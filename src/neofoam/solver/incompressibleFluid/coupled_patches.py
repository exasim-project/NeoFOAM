# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — ``@coupled`` reads the
# ``sends`` / ``receives`` annotations as live classes.

"""The coupled patches of the fluid's ``T``: the Dirichlet side on an OpenFOAM patch.

A patch of ``0/T`` is coupled when it names its condition in a ``coupled``
entry, as for ``heatConductionNeoN`` (the boundary defines the coupling and the
condition; ``system/regionCoupling.yaml`` only the numerics). On the OpenFOAM
side its ``type`` is ``fixedValue``, so the energy equation takes its values as
the wall temperature::

    interface
    {
        type        fixedValue;
        coupled     coupledTemperature;
        value       uniform 300;
    }

- ``coupledTemperature`` (:class:`CoupledTemperature`), the Dirichlet side, as
  ``heatConductionNeoN``'s: its faces take the partner's wall temperature, it
  sends the heat flux through them into the fluid,
  ``q = rho cp (nu/Pr + alphat) dT/dn`` (W/m^2, positive into the fluid), with
  ``rho`` and ``cp`` from the Boussinesq entries of ``constant/transportProperties``
  (:class:`~.models.boussinesq.BoussinesqConfig`).

The points are the face centres: the partner's faces need not match.

OpenFOAM's ``fixedValue`` drops the ``coupled`` entry when it writes ``T``; the
field writer adds it back (:class:`~.models.field_writer.CoupledEntriesWriteHook`),
so a written time restarts coupled.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pybFoam as pyf
from pybFoam import fvc
from pydantic import BaseModel

from neofoam.framework.context import Context
from neofoam.framework.coupling import Array, Coupling, coupled

from .models.boussinesq import BoussinesqConfig, boussinesq


class WallTemperature(BaseModel):
    """What the Neumann side sends: its wall temperature, per face (as ``heatConductionNeoN``'s)."""

    T: Array


class WallHeatFlux(BaseModel):
    """What the Dirichlet side sends: the heat flux through its faces into its region."""

    q: Array


def _boussinesq_config(ctx: Context) -> Optional[BoussinesqConfig]:
    """The Boussinesq config of the run, ``None`` without the boussinesq model."""
    for model in ctx.models.values():
        if getattr(model, "spec", None) is boussinesq:
            config: BoussinesqConfig = model.config
            return config
    return None


class FoamPatch:
    """A patch of the OpenFOAM field ``field``: its faces and their centres."""

    #: the name of its condition, its ``coupled`` entry in the field file
    condition: str = ""

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
        """Set the values of the patch.

        Through the zero-copy numpy view of the patch field: assigning a field
        to a ``fixedValue`` patch is a no-op in OpenFOAM (its ``operator=`` is
        disabled). They hold through ``correctBoundaryConditions`` and the solve.
        """
        np.asarray(self.ctx.fields[self.field][self.patch])[:] = np.asarray(values, dtype=float)


@coupled
class CoupledTemperature(FoamPatch):
    """The Dirichlet side: its faces take the partner's wall temperature, it sends its heat flux.

    Needs the boussinesq model (``T``, ``alphat``, ``Pr``) and ``rho`` and ``cp``
    in ``constant/transportProperties``; raises ``ValueError`` without them.
    """

    condition = "coupledTemperature"

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
        """The heat flux through each face into the fluid, ``rho cp (nu/Pr + alphat) dT/dn``.

        ``snGrad(T)`` on the patch is ``(T_wall - T_cell) deltaCoeffs``: positive
        when the wall is the hotter, as the heat flows into the fluid.
        """
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

#: The ``type`` a coupled patch of each condition has in OpenFOAM.
PATCH_TYPES: dict[str, str] = {
    "coupledTemperature": "fixedValue",
}


def coupled_entries(field: str, path: Path) -> dict[str, str]:
    """The coupled patches of the field file ``path``: patch -> its ``coupled`` condition.

    Raises ``ValueError`` for a condition it does not know or a patch of the wrong ``type``.
    """
    boundary = pyf.dictionary.read(str(path)).subDict("boundaryField")
    entries: dict[str, str] = {}
    for patch in map(str, boundary.toc()):
        entry = boundary.subDict(patch)
        if not entry.found("coupled"):
            continue
        name = str(entry.get_word("coupled"))
        if name not in COUPLED_PATCHES:
            raise ValueError(
                f"{path}: patch {patch} is coupled as {name!r}; known: {sorted(COUPLED_PATCHES)}"
            )
        patch_type = str(entry.get_word("type"))
        if patch_type != PATCH_TYPES[name]:
            raise ValueError(
                f"{path}: the coupled patch {patch} has type {patch_type}; a {name} patch of "
                f"{field} is '{PATCH_TYPES[name]}' (its values come from the coupling)"
            )
        entries[patch] = name
    return entries


def coupled_patches(field: str, ctx: Context, case_dir: Path) -> list[FoamPatch]:
    """The coupled patches of ``field``: the ``coupled`` entries of its start-time file.

    None when the case has no such file. Raises ``ValueError`` for a condition it
    does not know, or a coupled patch of the wrong ``type``. A coupled patch in a
    region the case does not couple raises from ``Coupling.add``.
    """
    path = case_dir / ctx.models["solution_loop"].timeName() / field
    if not path.is_file():
        return []
    return [
        COUPLED_PATCHES[name](field, patch, ctx)
        for patch, name in coupled_entries(field, path).items()
    ]
