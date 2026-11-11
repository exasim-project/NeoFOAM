# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — ``@coupled`` reads the
# ``sends`` / ``receives`` annotations as live classes.

"""The coupled patches of the fluid's ``T``: the Dirichlet side on an OpenFOAM ``fixedValue`` patch.

A patch of ``0/T`` is coupled by its entry ``coupled coupledTemperature;``.
"""

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from pybFoam import fvc

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling, coupled
from neofoam.solver.regions import WallHeatFlux, WallTemperature, coupled_entries

from .models.boussinesq import BoussinesqConfig, boussinesq


def _boussinesq_config(ctx: Context) -> Optional[BoussinesqConfig]:
    """The Boussinesq config of the run, ``None`` without the boussinesq model."""
    for model in ctx.models.values():
        if getattr(model, "spec", None) is boussinesq:
            config: BoussinesqConfig = model.config
            return config
    return None


class FoamPatch:
    """A patch of the OpenFOAM field ``field``: its faces and their centres."""

    def __init__(self, field: str, patch: str, ctx: Context) -> None:
        self.field = field
        self.patch = patch
        self.ctx = ctx
        mesh = ctx.fields[field].mesh()
        self.centres = np.array(np.asarray(mesh.Cf()[patch]))

    def wall_T(self) -> Any:
        """The values of the patch, a copy."""
        return np.array(np.asarray(self.ctx.fields[self.field][self.patch]))

    def set_wall_T(self, values: Any) -> None:
        # through the numpy view: OpenFOAM disables operator= of a fixedValue patch
        np.asarray(self.ctx.fields[self.field][self.patch])[:] = np.asarray(values, dtype=float)


@coupled
class CoupledTemperature(FoamPatch):
    """The Dirichlet side: takes the partner's wall temperature, sends its heat flux.

    Raises ``ValueError`` without the boussinesq model or its ``rho`` and ``cp``.
    """

    def __init__(self, field: str, patch: str, ctx: Context) -> None:
        config = _boussinesq_config(ctx)
        if config is None:
            raise ValueError(
                f"the coupled patch {patch} of {field} ({type(self).__name__}) needs the "
                "boussinesq model: beta and TRef in constant/transportProperties"
            )
        missing = [name for name in ("rho", "cp") if getattr(config, name) is None]
        if missing:
            raise ValueError(
                f"the coupled patch {patch} of {field} ({type(self).__name__}) exchanges the heat "
                f"flux rho cp (nu/Pr + alphat) dT/dn: set {' and '.join(missing)} (SI units) "
                "in constant/transportProperties"
            )
        super().__init__(field, patch, ctx)
        self.rho_cp = float(config.rho) * float(config.cp)  # type: ignore[arg-type]
        self.Pr = float(config.Pr)
        Coupling.of(ctx).add(self)

    def heat_flux(self) -> Any:
        """The heat flux through each face into the fluid, ``rho cp (nu/Pr + alphat) dT/dn``."""
        T = self.ctx.fields[self.field]
        sn_grad = fvc.snGrad(T)
        gradient = np.array(np.asarray(sn_grad()[self.patch]))
        nu = self.ctx.models["turbulence"].nu()
        nu_wall = np.array(np.asarray(nu()[self.patch]))
        alphat = np.array(np.asarray(self.ctx.fields["alphat"][self.patch]))
        return self.rho_cp * (nu_wall / self.Pr + alphat) * gradient

    @coupled.coordinates
    def points(self) -> Any:
        return self.centres

    @coupled.sends
    def send(self) -> WallHeatFlux:
        return WallHeatFlux(q=self.heat_flux())

    @coupled.receives
    def receive(self, data: WallTemperature) -> None:
        self.set_wall_T(data.T)


COUPLED_PATCHES: dict[str, type[FoamPatch]] = {
    "coupledTemperature": CoupledTemperature,
}

#: the ``type`` of a coupled patch of each condition, for ``coupled_entries``
PATCH_TYPES = {"coupledTemperature": "fixedValue"}


def start_entries(case_dir: Path, time_name: str) -> dict[str, str]:
    """The coupled entries of ``T`` at the start time; none for a fluid without ``T``."""
    path = case_dir / time_name / "T"
    return coupled_entries(path, PATCH_TYPES) if path.is_file() else {}


def coupled_patches(field: str, ctx: Context, entries: Mapping[str, str]) -> list[FoamPatch]:
    """The coupled patches of ``field``; ``entries``: patch -> condition (``coupled_entries``)."""
    return [COUPLED_PATCHES[condition](field, patch, ctx) for patch, condition in entries.items()]
