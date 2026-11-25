# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver
# matches ``Annotated[..., "models"]`` live.

"""Boussinesq buoyancy for incompressibleFluidNeoN, as OpenFOAM's buoyantBoussinesqPimpleFoam.

Use it for a natural-convection case: it is *detected* from ``beta`` and ``TRef`` in
``constant/transportProperties``, with ``constant/g``, ``0/T``, ``0/p_rgh`` and
``0/alphat``; the files of the native solver,
read the same way as by the pybFoam twin (:mod:`neofoam.solver.incompressibleFluid.models.
boussinesq`, whose configs and detection it shares). It adds

* the fields ``T`` and ``p_rgh`` (solved), ``alphat`` (``nut/Prt``,
  ``alphatJayatillekeWallFunction`` walls evaluated here), ``rhok = 1 - beta (T - TRef)``
  and the geopotential ``gh``/``ghf``; ``p = p_rgh + rhok gh`` is computed;
* per outer corrector, after the momentum predictor and before the pressure correction
  (the order of the native solver: U, T, rhok, p_rgh): the energy equation with
  ``alphaEff = nu/Pr + alphat`` and the ``rhok`` update.

The momentum predictor and the ``p_rgh`` corrector are the ``*_boussinesq`` arms of the
PIMPLE algorithm, switched on in ``@resolve``. ``prghPressure`` patches of ``p_rgh`` get
``p - rhok gh`` after every ``rhok`` update. Not supported: SIMPLE, fvOptions, MRF and a
moving mesh together with buoyancy, and a coupled region (``0/T`` is read from the working
directory, as the rest of the NeoN solver reads its case). No ``Model.field`` declarations:
on an optional model they would be synthesised as pybFoam reads.

Example (``constant/transportProperties``)::

    nu 1e-05; beta 3e-03; TRef 300; Pr 0.7; Prt 0.85;
"""

import ctypes
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import neon._neon as nn  # NeoN Python bindings
import numpy as np
import pybFoam as pyf
from foamlib import FoamFieldFile

from neofoam import neofoam_bindings as nfb
from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import ConfigContext, field, lazy, model
from neofoam.solver.incompressibleFluid.models.boussinesq import (
    BoussinesqConfig,
    GravityConfig,
    detect_model,
)

from .incompressibleFluidNeoNModel import Model, incompressibleFluidNeoNModel

__all__ = ["ThermalDiffusivity", "boussinesqNeoN"]

# The nutWallFunction coefficients the Jayatilleke wall function reads (wallFunctionCoefficients
# defaults; a case that overrides Cmu/kappa/E on its nut patches is not supported).
_CMU = 0.09
_KAPPA = 0.41
_E = 9.8
# The PIMPLE spec's name; not imported, since the algorithm imports this module's slices.
_PIMPLE = "PimpleNeoN"

boussinesqNeoN = (
    Model("boussinesq").register_with(incompressibleFluidNeoNModel).labeled("Buoyancy (Boussinesq)")
)
boussinesqNeoN.config(BoussinesqConfig)
boussinesqNeoN.config(GravityConfig)
BoussinesqNeoNFvSchemes = boussinesqNeoN.config(fvSchemes)
BoussinesqNeoNFvSolution = boussinesqNeoN.config(fvSolution)

boussinesqNeoN.detect(detect_model)


@boussinesqNeoN.load
def load(case_dir: Path, _instance_id: str) -> BoussinesqConfig:
    """``beta``, ``TRef``, ``Pr``, ``Prt`` of ``constant/transportProperties``."""
    return BoussinesqConfig.load(case_dir=case_dir)


@boussinesqNeoN.resolve
def resolve(_config: BoussinesqConfig, ctx: ConfigContext) -> None:
    """Switch the PIMPLE algorithm to its ``p_rgh`` arms."""
    if not ctx.contains(_PIMPLE):
        raise ValueError(
            "incompressibleFluidNeoN: Boussinesq buoyancy needs the transient PIMPLE algorithm"
        )
    pimple = ctx.get(_PIMPLE)
    pimple.config = replace(pimple.config, use_boussinesq=True)


class ThermalDiffusivity:
    """``alphaEff = nu/Pr + alphat`` on the faces, with ``alphat = nut/Prt``.

    As the TEqn.H of buoyantBoussinesqPimpleFoam, called once per energy solve: ``alphat``
    takes the turbulence model's current ``nut``, then its
    ``alphatJayatillekeWallFunction`` faces take the thermal wall law (from the near-wall
    ``k``, so zero for a laminar model). Not thread safe.
    """

    def __init__(
        self,
        runtime: Any,
        config: BoussinesqConfig,
        turbulence: Any,
        alphat: Any,
    ) -> None:
        self._config = config
        self._turbulence = turbulence
        self._alphat = alphat
        self._nu = float(nfb.read_transport_viscosity(runtime))
        self._interpolation = _linear_interpolation(runtime)
        self._walls = _jayatilleke_walls(runtime)
        boundary = runtime.nf_mesh.boundary_mesh()
        self._owners = _host(boundary.face_owners())
        self._wall_distance = _host(
            nfb.build_near_wall_dist(runtime, pyf.nearWallDist(runtime.mesh)).boundary_data_value()
        )

    def __call__(self) -> Any:
        """Update ``alphat`` and return the face field ``alphaEff``."""
        self._alphat.assign(self._turbulence.nut() / self._config.Prt)
        if self._walls and self._turbulence.has_field("k"):
            self._set_wall_alphat(_host(self._turbulence.field("k").internal_vector()))
        alpha_eff = self._interpolation.interpolate(self._alphat + self._nu / self._config.Pr)
        alpha_eff.name = "alphaEff"
        return alpha_eff

    def _set_wall_alphat(self, k: np.ndarray) -> None:
        """alphatJayatillekeWallFunction::updateCoeffs on every such patch."""
        for patch_id, faces, prt in self._walls:
            y_plus = (
                _CMU**0.25 * self._wall_distance[faces] * np.sqrt(k[self._owners[faces]]) / self._nu
            )
            # the patch is calculated, so the imposed values stay until the next update
            self._alphat.set_patch_fixed_values(
                patch_id, _jayatilleke_alphat(y_plus, self._nu, self._config.Pr, prt)
            )


def _jayatilleke_alphat(y_plus: np.ndarray, nu: float, pr: float, prt: float) -> np.ndarray:
    """The wall alphat of the Jayatilleke thermal law, zero below the thermal y+."""
    ratio = pr / prt
    p = 9.24 * (ratio**0.75 - 1.0) * (1.0 + 0.28 * np.exp(-0.007 * ratio))
    y_plus_therm = _thermal_y_plus(ratio, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = nu * (y_plus / (prt * (np.log(_E * y_plus) / _KAPPA + p)) - 1.0 / pr)
    return np.where(y_plus > y_plus_therm, np.maximum(0.0, kt), 0.0)


def _thermal_y_plus(ratio: float, p: float) -> float:
    """alphatJayatillekeWallFunction::yPlusTherm: the Newton iteration of OpenFOAM."""
    y_plus = 11.0
    for _ in range(10):
        f = y_plus - (np.log(_E * y_plus) / _KAPPA + p) / ratio
        df = 1.0 - 1.0 / (y_plus * _KAPPA * ratio)
        y_plus_new = y_plus - f / df
        if abs(y_plus_new - y_plus) < 0.01:
            return float(y_plus_new)
        y_plus = y_plus_new
    return float(y_plus)


def _patches_of_type(runtime: Any, field_name: str, patch_type: str) -> list[tuple[int, Any]]:
    """The id and dictionary of every patch of ``0/<field_name>`` with type ``patch_type``."""
    boundary_field = FoamFieldFile(Path("0", field_name)).boundary_field
    boundary = runtime.mesh.boundary()
    patches = []
    for patch_id in range(len(runtime.nf_mesh.boundary_mesh().offset()) - 1):
        entry = _patch_entry(boundary_field, str(boundary[patch_id].name()))
        if entry is not None and entry["type"] == patch_type:
            patches.append((patch_id, entry))
    return patches


def _patch_entry(boundary_field: Any, patch: str) -> Any:
    """The boundaryField entry of ``patch``: its name, else the first matching regex key."""
    if patch in boundary_field:
        return boundary_field[patch]
    for key in boundary_field:
        if re.fullmatch(str(key).strip('"'), patch):
            return boundary_field[key]
    return None


def _jayatilleke_walls(runtime: Any) -> list[tuple[int, slice, float]]:
    """The id, boundary faces and ``Prt`` of every ``alphatJayatillekeWallFunction`` patch."""
    offset = runtime.nf_mesh.boundary_mesh().offset()
    return [
        (patch_id, slice(offset[patch_id], offset[patch_id + 1]), float(entry.get("Prt", 0.85)))
        for patch_id, entry in _patches_of_type(runtime, "alphat", "alphatJayatillekeWallFunction")
    ]


def _prgh_pressure_patches(runtime: Any) -> list[tuple[int, float]]:
    """The patch id and ``p`` of every ``prghPressure`` patch of ``p_rgh``."""
    return [
        (patch_id, _uniform(entry["p"]))
        for patch_id, entry in _patches_of_type(runtime, "p_rgh", "prghPressure")
    ]


def _uniform(value: Any) -> float:
    """The value of a ``uniform <scalar>`` entry."""
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"prghPressure: only a uniform scalar p is supported, got {value!r}")


def impose_prgh_pressure(p_rgh: Any, rhok: Any, gh: Any, patches: list[tuple[int, float]]) -> None:
    """Set each prghPressure patch of ``p_rgh`` to ``p - rhok*gh`` with the current ``rhok``."""
    offset = p_rgh.mesh().boundary_mesh().offset()
    rhok_b = _host(rhok.boundary_data_value())
    gh_b = _host(gh.boundary_data_value())
    for patch_id, p in patches:
        faces = slice(offset[patch_id], offset[patch_id + 1])
        p_rgh.set_patch_fixed_values(
            patch_id, np.ascontiguousarray(p - rhok_b[faces] * gh_b[faces])
        )


@boussinesqNeoN.build
def build(config: BoussinesqConfig) -> list[Any]:
    """The Boussinesq fields and the thermal diffusivity the energy equation solves with.

    In createFields.H order: ``T``, ``p_rgh``, ``alphat``, ``rhok``, ``gh``/``ghf``, then
    ``p = p_rgh + rhok gh``, shifted to ``pRefValue`` in the reference cell when ``p_rgh``
    needs a reference (``p_rgh`` itself is not shifted at start, as in the native solver).
    """

    def read(name: str) -> Any:
        def create(ctx: dict[str, Any]) -> Any:
            _load_generic_patch_fields()
            return nfb.read_scalar_volume_field(ctx["_neon_runtime"], name)

        return create

    def create_rhok(ctx: dict[str, Any]) -> Any:
        rhok = nfb.create_uniform_volume_field(ctx["_neon_runtime"], "rhok", 1.0)
        update_rhok_field(rhok, ctx["fields.T"], config)
        return rhok

    def create_prgh_patches(ctx: dict[str, Any]) -> list[tuple[int, float]]:
        # OpenFOAM evaluates a prghPressure patch first in the pressure equation, so until
        # then it holds the value of the file; NeoN needs it imposed to treat it as fixed.
        p_rgh = ctx["fields.p_rgh"]
        patches = _prgh_pressure_patches(ctx["_neon_runtime"])
        offset = p_rgh.mesh().boundary_mesh().offset()
        values = _host(p_rgh.boundary_data_value())
        for patch_id, _ in patches:
            faces = values[offset[patch_id] : offset[patch_id + 1]]
            p_rgh.set_patch_fixed_values(patch_id, np.ascontiguousarray(faces))
        return patches

    def create_diffusivity(ctx: dict[str, Any]) -> ThermalDiffusivity:
        return ThermalDiffusivity(
            ctx["_neon_runtime"], config, ctx["models.turbulence"], ctx["fields.alphat"]
        )

    def initialise_p(ctx: dict[str, Any]) -> None:
        p = ctx["fields.p"]
        p.assign(ctx["fields.p_rgh"] + ctx["fields.rhok"] * ctx["fields.gh"])
        reference = ctx["models.pressure_reference"]
        if reference.needs_ref:
            p.assign(p + (reference.value - _host(p.internal_vector())[reference.cell]))

    return [
        field("T", read("T"), depends_on=["_neon_runtime"], write=True),
        field("p_rgh", read("p_rgh"), depends_on=["_neon_runtime"], write=True),
        field("alphat", read("alphat"), depends_on=["_neon_runtime"], write=True),
        field("rhok", create_rhok, depends_on=["_neon_runtime", "fields.T"]),
        field("gh", lambda ctx: nfb.create_gh(ctx["_neon_runtime"]), depends_on=["_neon_runtime"]),
        field(
            "ghf", lambda ctx: nfb.create_ghf(ctx["_neon_runtime"]), depends_on=["_neon_runtime"]
        ),
        model("mag_sf", lambda ctx: nfb.mag_sf(ctx["_neon_runtime"]), depends_on=["_neon_runtime"]),
        model(
            "prgh_pressure_patches",
            create_prgh_patches,
            depends_on=["_neon_runtime", "fields.p_rgh", "fields.rhok", "fields.gh"],
        ),
        model(
            "thermal_diffusivity",
            create_diffusivity,
            depends_on=["_neon_runtime", "models.turbulence", "fields.alphat"],
        ),
        lazy(
            "p_hydrostatic",
            initialise_p,
            depends_on=[
                "fields.p",
                "fields.p_rgh",
                "fields.rhok",
                "fields.gh",
                "models.pressure_reference",
            ],
        ),
    ]


def update_rhok_field(rhok: Any, T: Any, config: BoussinesqConfig) -> None:
    """``rhok = 1 - beta*(T - TRef)``, in the native solver's operation order."""
    rhok.assign(1.0 - (T - config.TRef) * config.beta)


@boussinesqNeoN.operation(operation_number="2.5", depends_on=["momentum"])
@BoussinesqNeoNFvSchemes.add(ddt="default", div="div(phi,T)", laplacian="default")
@BoussinesqNeoNFvSolution.add("T")
def solve_energy(
    T: Any,
    phi: Any,
    thermal_diffusivity: Annotated[Any, "models"],
    pimple_state: Annotated[Any, "models"],
    neon_runtime: Annotated[Any, "models"],
) -> FieldUpdates:
    """TEqn.H: the energy equation with this outer corrector's flux (``TFinal`` on the last)."""
    equation = nn.imp.ddt(T) + nn.imp.div(phi, T) - nn.imp.laplacian(thermal_diffusivity(), T)
    solver = nfb.PDESolverScalar(equation, T, neon_runtime)
    solver.set_final_iter(bool(pimple_state.control.final_iter()))
    solver.solve()
    # NeoN's solve, unlike OpenFOAM's, leaves a zeroGradient patch at its old values,
    # which rhok reads
    T.correct_boundary_conditions()
    return FieldUpdates({"T": T})


@boussinesqNeoN.operation(operation_number="2.7", depends_on=["solve_energy"])
def update_rhok(
    self: Any,
    T: Any,
    rhok: Any,
    p_rgh: Any,
    gh: Any,
    prgh_pressure_patches: Annotated[Any, "models"],
) -> None:
    """The end of TEqn.H: ``rhok`` from the new ``T``; the prghPressure patches follow it."""
    update_rhok_field(rhok, T, self.config)
    impose_prgh_pressure(p_rgh, rhok, gh, prgh_pressure_patches)


def _linear_interpolation(runtime: Any) -> Any:
    """The linear face interpolation of TEqn.H's ``laplacian(alphaEff, T)``."""
    return nn.SurfaceInterpolationScalar(
        runtime.executor, runtime.nf_mesh, nn.TokenList(["linear"])
    )


def _load_generic_patch_fields() -> None:
    """Load OpenFOAM's generic patch fields, so ``0/alphat`` may name a compressible wall law.

    OpenFOAM reads a patch type it does not know (with its value) only through them;
    NeoFOAM's reader then maps it to calculated. Loading twice is a no-op.
    """
    library = Path(os.environ["FOAM_LIBBIN"]) / "libgenericPatchFields.so"
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _host(vector: Any) -> np.ndarray:
    """A host copy of a NeoN vector as numpy."""
    return np.asarray(vector.copy_to_host())
