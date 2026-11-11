# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Coupled patches of ``T``: a Dirichlet-Neumann pair on ``calculated`` NeoN patches."""

from __future__ import annotations

from typing import Any, Mapping

import neon._neon as nn
import numpy as np

from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import Context
from neofoam.framework.coupling import coupled
from neofoam.solver.regions import WallHeatFlux, WallTemperature


def _host(vector: Any) -> Any:
    """A NeoN vector as a numpy array on the host."""
    return np.asarray(vector.copy_to_host())


class NeoNPatch:
    """A patch of the NeoN field ``field``: its faces, their cells and the conductance between."""

    def __init__(self, field: str, patch: str, ctx: Context) -> None:
        rt = ctx.models["neon_runtime"]
        self.field = field
        self.patch = patch
        self.ctx = ctx
        self.patch_id = int(rt.mesh.boundary().findPatchID(patch))
        boundary = rt.nf_mesh.boundary_mesh()
        offset = boundary.offset()
        self.faces = slice(offset[self.patch_id], offset[self.patch_id + 1])
        self.owners = _host(boundary.face_owners())[self.faces]
        self.areas = _host(boundary.face_areas())[self.faces]
        # per face area
        self.h = ctx.models["conduction"].properties.k * _host(boundary.delta_coeffs())[self.faces]

    def cell_T(self) -> Any:
        return _host(self.ctx.fields[self.field].internal_vector())[self.owners]

    def wall_T(self) -> Any:
        return _host(self.ctx.fields[self.field].boundary_data_value())[self.faces]

    def set_wall_T(self, values: Any) -> None:
        self.ctx.fields[self.field].set_patch_fixed_values(
            self.patch_id, np.ascontiguousarray(values, dtype=float)
        )

    def terms(self) -> list[Any]:
        """The terms the patch adds to the right-hand side."""
        return []

    def solved(self) -> None:
        """Called after the field is solved and its boundary corrected."""


@coupled
class CoupledTemperature(NeoNPatch):
    """The Dirichlet side: takes the partner's wall temperature, sends its heat flux."""

    def __init__(self, field: str, patch: str, ctx: Context) -> None:
        super().__init__(field, patch, ctx)
        self.set_wall_T(self.wall_T())

    @coupled.sends
    def send(self) -> WallHeatFlux:
        return WallHeatFlux(q=self.h * (self.wall_T() - self.cell_T()))

    @coupled.receives
    def receive(self, data: WallTemperature) -> None:
        self.set_wall_T(data.T)


@coupled
class CoupledHeatFlux(NeoNPatch):
    """The Neumann side: takes the partner's heat flux, sends its wall temperature.

    NeoN sets no per-face gradient from Python, so the heat flux enters as an explicit
    source in the cells next to the faces.
    """

    def __init__(self, field: str, patch: str, ctx: Context) -> None:
        super().__init__(field, patch, ctx)
        rt = ctx.models["neon_runtime"]
        self.capacities = ctx.models["conduction"].rho_cp * _host(rt.nf_mesh.cell_volumes)
        self.source = nfb.create_uniform_volume_field(rt, f"heatIn({patch})", 0.0)
        # until the partner sends, the heat flux its value in 0/T gives
        self._set_heat_in(self.h * (super().wall_T() - self.cell_T()))

    def _set_heat_in(self, heat_in: Any) -> None:
        self.heat_in = np.asarray(heat_in, dtype=float)
        source = np.zeros(len(self.capacities))
        np.add.at(source, self.owners, self.heat_in * self.areas)
        nfb.copy_from_host(self.source, source / self.capacities)

    def wall_T(self) -> Any:
        return self.cell_T() + self.heat_in / self.h

    def terms(self) -> list[Any]:
        return [nn.exp.source(self.source)]

    def solved(self) -> None:
        """Store the sent wall temperature as the face values, so the written ``T`` holds it."""
        # face by face: the only write into part of the boundary values NeoN binds on any
        # executor; refValue / refGrad stay, the equation reads the flux from the source
        values = self.ctx.fields[self.field].boundary_data_value()
        for face, value in zip(range(self.faces.start, self.faces.stop), self.wall_T()):
            nn.fill_range(values, float(value), face, face + 1)

    @coupled.sends
    def send(self) -> WallTemperature:
        return WallTemperature(T=self.wall_T())

    @coupled.receives
    def receive(self, data: WallHeatFlux) -> None:
        self._set_heat_in(-np.asarray(data.q, dtype=float))


COUPLED_PATCHES: dict[str, type[NeoNPatch]] = {
    "coupledTemperature": CoupledTemperature,
    "coupledHeatFlux": CoupledHeatFlux,
}

#: the ``type`` of a coupled patch of each condition, for ``coupled_entries``
PATCH_TYPES = {condition: "calculated" for condition in COUPLED_PATCHES}


def coupled_patches(field: str, ctx: Context, entries: Mapping[str, str]) -> dict[str, NeoNPatch]:
    """The coupled patches of ``field`` by patch; ``entries``: patch -> condition."""
    return {
        patch: COUPLED_PATCHES[condition](field, patch, ctx) for patch, condition in entries.items()
    }
