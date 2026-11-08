# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — ``@coupled`` reads the
# ``sends`` / ``receives`` annotations as live classes.

"""The coupled patches of ``T``: a Dirichlet-Neumann pair on NeoN patches.

A patch of ``0/T`` is coupled when it names its condition in a ``coupled``
entry; its ``type`` is ``calculated``, so NeoN leaves its values to the
coupling (``correct_boundary_conditions`` keeps them)::

    interface
    {
        type        calculated;
        coupled     coupledTemperature;   // or coupledHeatFlux
        value       uniform 300;
    }

- ``coupledTemperature`` (:class:`CoupledTemperature`), the Dirichlet side: its
  faces take the partner's wall temperature (``set_patch_fixed_values``), it
  sends the heat flux through them into its own region.
- ``coupledHeatFlux`` (:class:`CoupledHeatFlux`), the Neumann side: the heat
  flux into the partner leaves its own region, as an explicit source in the
  cells next to its faces (NeoN sets no per-face gradient from Python); it sends
  its wall temperature.

The heat flux is per face area, ``k dT/dn``. The points are the face centres:
the partner's faces need not match.
"""

from pathlib import Path
from typing import Any

import neon._neon as nn  # NeoN Python bindings
import numpy as np
import pybFoam as pyf
from pydantic import BaseModel

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.framework.context import Context
from neofoam.framework.coupling import Array, Coupling, coupled


def _host(vector: Any) -> Any:
    """A NeoN vector as a numpy array on the host."""
    return np.asarray(vector.copy_to_host())


class WallTemperature(BaseModel):
    """What the Neumann side sends: its wall temperature, per face."""

    T: Array


class WallHeatFlux(BaseModel):
    """What the Dirichlet side sends: the heat flux through its faces into its region."""

    q: Array


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
        self.centres = _host(boundary.face_centers())[self.faces]
        # the conductance between a face and its cell, per face area
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
        """What the patch adds to the right-hand side of the equation of its field."""
        return []


@coupled
class CoupledTemperature(NeoNPatch):
    """The Dirichlet side: its faces take the partner's wall temperature, it sends its heat flux."""

    def __init__(self, field: str, patch: str, ctx: Context) -> None:
        super().__init__(field, patch, ctx)
        self.set_wall_T(self.wall_T())  # fixed at its value in 0/T until the partner sends
        Coupling.of(ctx).add(self)

    @coupled.coordinates
    def points(self) -> Any:
        return self.centres

    @coupled.sends
    def send(self) -> WallHeatFlux:
        return WallHeatFlux(q=self.h * (self.wall_T() - self.cell_T()))

    @coupled.receives
    def receive(self, data: WallTemperature) -> None:
        self.set_wall_T(data.T)


@coupled
class CoupledHeatFlux(NeoNPatch):
    """The Neumann side: the heat flux into the partner's region leaves its own; it sends its wall
    temperature."""

    def __init__(self, field: str, patch: str, ctx: Context) -> None:
        super().__init__(field, patch, ctx)
        rt = ctx.models["neon_runtime"]
        # dT/dt gains q A / (rho cp V) in a cell from the heat flux q through its face
        self.capacities = ctx.models["conduction"].rho_cp * _host(rt.nf_mesh.cell_volumes)
        # the heat flux through each face into the region; its source in the cells
        self.heat_in = np.zeros(len(self.owners))
        self.source = nfb.create_uniform_volume_field(rt, f"heatIn({patch})", 0.0)
        Coupling.of(ctx).add(self)

    def wall_T(self) -> Any:
        return self.cell_T() + self.heat_in / self.h

    def terms(self) -> list[Any]:
        return [nn.exp.source(self.source)]

    @coupled.coordinates
    def points(self) -> Any:
        return self.centres

    @coupled.sends
    def send(self) -> WallTemperature:
        return WallTemperature(T=self.wall_T())

    @coupled.receives
    def receive(self, data: WallHeatFlux) -> None:
        self.heat_in = -np.asarray(data.q, dtype=float)
        source = np.zeros(len(self.capacities))
        np.add.at(source, self.owners, self.heat_in * self.areas)
        nfb.copy_from_host(self.source, source / self.capacities)


COUPLED_PATCHES: dict[str, type[NeoNPatch]] = {
    "coupledTemperature": CoupledTemperature,
    "coupledHeatFlux": CoupledHeatFlux,
}


def coupled_patches(field: str, ctx: Context, case_dir: Path) -> list[NeoNPatch]:
    """The coupled patches of ``field``: the ``coupled`` entries of its start-time file.

    Raises ``ValueError`` for a condition it does not know, or a coupled patch not
    ``calculated``. A coupled patch in a region the case does not couple raises
    from ``Coupling.add``.
    """
    path = case_dir / ctx.models["solution_loop"].timeName() / field
    boundary = pyf.dictionary.read(str(path)).subDict("boundaryField")
    patches: list[NeoNPatch] = []
    for patch in map(str, boundary.toc()):
        entry = boundary.subDict(patch)
        if not entry.found("coupled"):
            continue
        name = str(entry.get_word("coupled"))
        if name not in COUPLED_PATCHES:
            raise ValueError(
                f"{path}: patch {patch} is coupled as {name!r}; known: {sorted(COUPLED_PATCHES)}"
            )
        if str(entry.get_word("type")) != "calculated":
            raise ValueError(
                f"{path}: the coupled patch {patch} has type {entry.get_word('type')}; a coupled "
                "patch is 'calculated' (its values come from the coupling)"
            )
        patches.append(COUPLED_PATCHES[name](field, patch, ctx))
    return patches
