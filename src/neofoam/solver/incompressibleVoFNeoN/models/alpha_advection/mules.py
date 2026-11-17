# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the dependency_resolver
# matches ``Annotated[..., "models"]`` live.

"""MULES alpha advection on NeoN: interFoam's alphaEqnSubCycle.H plus the mixture update."""

from dataclasses import dataclass
from typing import Annotated, Any, Optional

import neon._neon as nn
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.algorithms.mules import (
    AlphaControls,
    AlphaFluxes,
    PhaseFractions,
    VolumetricFlux,
    read_alpha_controls,
    read_alpha_ddt_off_centring,
    sub_cycled_alpha_eqn,
)
from neofoam.algorithms.mules._neon_ops import NeoNMulesOps
from neofoam.fields import (
    CyclicBC,
    EmptyBC,
    FixedValueBC,
    GenericBC,
    InletOutletBC,
    Scalar,
    SymmetryBC,
    SymmetryPlaneBC,
    ZeroGradientBC,
)
from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.model import Model
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata
from neofoam.io import DictFile

__all__ = ["MulesNeoNState", "mulesNeoN"]

mulesNeoN = Model("MULESNeoN").labeled("MULES")

# Schema-only, as on the pybFoam MULES spec: surfaced through configurations(solver).
MulesFvSchemes = mulesNeoN.config(fvSchemes)
MulesFvSolution = mulesNeoN.config(fvSolution)

mulesNeoN.field(
    "alpha.water",
    dimensions=[0, 0, 0, 0, 0, 0, 0],
    value_type=Scalar,
    allowed_bcs=[
        FixedValueBC,
        ZeroGradientBC,
        InletOutletBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    write=True,
)

# The schemes the NeoN MULES kernels hard-code, keyed by their fvSchemes entry.
_NEON_KERNEL_SCHEMES = {
    ("divSchemes", "div(phi,alpha)"): "Gauss vanLeer",
    ("divSchemes", "div(phirb,alpha)"): "Gauss linear",
}


@dataclass
class MulesNeoNState:
    """The alpha step's controls, ops and phase fluxes, built once for the run.

    Use it through the ``mules_state`` model of :data:`mulesNeoN`; the pybFoam solver keeps
    its own state. The controls are read once: unlike alphaEqn.H, a changed ``fvSolution`` is
    not picked up during the run. ``rho_phi_sum`` is only allocated when sub-cycling. Not
    thread-safe: one time loop owns it.

    Example::

        def operation(mules_state: Annotated[MulesNeoNState, "models"]) -> None: ...
    """

    ops: NeoNMulesOps
    controls: AlphaControls
    oc_coeff: float
    fluxes: AlphaFluxes
    rho1: float
    rho2: float
    nu1: float
    nu2: float
    rho_phi_sum: Optional[Any] = None


def _read_kernel_controls(alpha_name: str) -> tuple[float, int]:
    """``cAlpha`` (required) and ``nLimiterIter`` (MULES default 3) of the alpha solver dict."""
    solvers = pyf.dictionary.read("system/fvSolution").subDict("solvers")
    alpha_dict = solvers.subDict(alpha_name)
    return (
        float(alpha_dict.get[float]("cAlpha")),
        int(alpha_dict.getOrDefault[int]("nLimiterIter", 3)),
    )


def _check_kernel_schemes() -> None:
    """Refuse a case whose alpha schemes differ from the ones the NeoN kernels implement."""
    schemes = DictFile("system/fvSchemes")
    for key, supported in _NEON_KERNEL_SCHEMES.items():
        found = schemes.get[str](key) if schemes.found(key) else None
        if found != supported:
            raise NotImplementedError(
                f"incompressibleVoFNeoN: {'/'.join(key)} is {found!r}, but the NeoN MULES "
                f"kernels implement only {supported!r}"
            )


@mulesNeoN.build
def build(self: Any) -> list[Any]:
    """alpha1, phi, rho, the mixture nu and rhoPhi of createFields.H, and the MULES state."""

    def create_alpha1(ctx: dict[str, Any]) -> Any:
        phase1 = ctx["models.phase_properties"]["phase1"]
        return nfb.read_scalar_volume_field(ctx["_neon_runtime"], f"alpha.{phase1}")

    def create_phi(ctx: dict[str, Any]) -> Any:
        return nfb.create_phi(ctx["_neon_runtime"], "U")

    def create_rho(ctx: dict[str, Any]) -> Any:
        properties = ctx["models.phase_properties"]
        rho = nfb.create_registered_scalar_volume_field(ctx["_neon_runtime"], "rho", 0.0)
        nfb.update_mixture_density(
            ctx["fields.alpha1"], properties["rho1"], properties["rho2"], rho
        )
        return rho

    def create_nu(ctx: dict[str, Any]) -> Any:
        properties = ctx["models.phase_properties"]
        nu = nfb.create_uniform_volume_field(ctx["_neon_runtime"], "nu", 0.0)
        nfb.update_mixture_viscosity(
            ctx["fields.alpha1"],
            properties["rho1"],
            properties["nu1"],
            properties["rho2"],
            properties["nu2"],
            nu,
        )
        return nu

    def create_rho_phi(ctx: dict[str, Any]) -> Any:
        # createFields.H: interpolate(rho)*phi of the flux as read, before any projection.
        # The name is the div(rhoPhi,U) scheme key of the momentum equation.
        rho_phi = nfb.linear_interpolate(ctx["_neon_runtime"], ctx["fields.rho"])
        nfb.multiply(rho_phi, ctx["fields.phi"], rho_phi)
        rho_phi.name = "rhoPhi"
        return rho_phi

    def create_state(ctx: dict[str, Any]) -> MulesNeoNState:
        alpha1 = ctx["fields.alpha1"]
        phi = ctx["fields.phi"]
        runtime = ctx["_neon_runtime"]
        properties = ctx["models.phase_properties"]
        c_alpha, n_limiter_iter = _read_kernel_controls(alpha1.name)
        if c_alpha != 0.0:
            _check_kernel_schemes()
        controls = AlphaControls(*read_alpha_controls(alpha1.name))
        return MulesNeoNState(
            # The compression reads the ops' own nHatf; surface tension reads models.interface.
            ops=NeoNMulesOps(ctx["_foam_time"], runtime, alpha1, phi, (c_alpha, n_limiter_iter)),
            controls=controls,
            oc_coeff=read_alpha_ddt_off_centring(runtime.mesh, controls.n_alpha_sub_cycles),
            # createAlphaFluxes.H's persistent phase fluxes, overwritten every corrector.
            fluxes=AlphaFluxes(phi * 0.0, phi * 0.0),
            rho1=properties["rho1"],
            rho2=properties["rho2"],
            nu1=properties["nu1"],
            nu2=properties["nu2"],
        )

    return [
        model(
            "phase_properties",
            lambda ctx: nfb.read_two_phase_transport_properties(ctx["_neon_runtime"]),
            depends_on=["_neon_runtime"],
        ),
        field(
            "alpha1",
            create_alpha1,
            depends_on=["_neon_runtime", "models.phase_properties"],
            write=True,
        ),
        field("phi", create_phi, depends_on=["_neon_runtime", "fields.U"]),
        field("rho", create_rho, depends_on=["fields.alpha1", "models.phase_properties"]),
        field("nu", create_nu, depends_on=["fields.alpha1", "models.phase_properties"]),
        field("rhoPhi", create_rho_phi, depends_on=["fields.rho", "fields.phi"]),
        model(
            "mules_state",
            create_state,
            depends_on=["_foam_time", "fields.alpha1", "fields.phi", "models.phase_properties"],
        ),
    ]


def _rho_phi_of_pass(state: MulesNeoNState, alpha_phi10: Any, phi_cn: Any, rho_phi: Any) -> None:
    """alphaEqn.H's ``rhoPhi = alphaPhi10*(rho1 - rho2) + phiCN*rho2`` (Euler ``ddt(rho,U)``)."""
    nfb.linear_combination(state.rho1 - state.rho2, alpha_phi10, state.rho2, phi_cn, rho_phi)


def _advance_alpha(state: MulesNeoNState, alpha1: Any, phi: Any, rhoPhi: Any) -> None:
    """alphaEqnSubCycle.H without the final ``rho`` update: alpha1 and rhoPhi."""
    flux = VolumetricFlux(phi, state.ops.crank_nicolson_flux(state.oc_coeff))
    passes = sub_cycled_alpha_eqn(
        state.ops, PhaseFractions(alpha1), flux, state.controls, state.fluxes
    )
    if state.controls.n_alpha_sub_cycles == 1:
        for _, alpha_phi10 in passes:
            _rho_phi_of_pass(state, alpha_phi10, flux.phi_cn, rhoPhi)
        return

    if state.rho_phi_sum is None:
        state.rho_phi_sum = rhoPhi * 0.0
    state.rho_phi_sum.assign(rhoPhi * 0.0)
    for fraction, alpha_phi10 in passes:
        _rho_phi_of_pass(state, alpha_phi10, flux.phi_cn, rhoPhi)
        nfb.linear_combination(1.0, state.rho_phi_sum, fraction, rhoPhi, state.rho_phi_sum)
    rhoPhi.assign(state.rho_phi_sum)


@mulesNeoN.operation(operation_number="2.0")
@MulesFvSchemes.add(ddt="ddt(alpha)", div=["div(phi,alpha)", "div(phirb,alpha)"])
@MulesFvSolution.add("alpha.water")
def alpha_advection(
    alpha1: Any,
    phi: Any,
    rho: Any,
    nu: Any,
    rhoPhi: Any,
    mules_state: Annotated[MulesNeoNState, "models"],
) -> FieldUpdates:
    """Advance alpha1 by MULES, then rhoPhi, rho and the mixture nu (interFoam.C's alpha block)."""
    # ddt(rho, U) reads rho.oldTime(): the density at the start of this time step.
    nn.rotate_old_times(rho)
    _advance_alpha(mules_state, alpha1, phi, rhoPhi)
    nfb.update_mixture_density(alpha1, mules_state.rho1, mules_state.rho2, rho)
    # mixture.correct() after alphaEqnSubCycle.H: calcNu on the new alpha1.
    nfb.update_mixture_viscosity(
        alpha1, mules_state.rho1, mules_state.nu1, mules_state.rho2, mules_state.nu2, nu
    )
    return FieldUpdates({"alpha1": alpha1, "rho": rho, "nu": nu, "rhoPhi": rhoPhi})


@mulesNeoN.operation_collection
def collected_operations(self: Any) -> Operations:
    """Expose ``alpha_advection`` (the collection path bypasses the default wrapping)."""
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=SequentialOp(mulesNeoN.wrap_operation(alpha_advection, self)),
            metadata=OperationMetadata(
                op_name="alpha_advection", depends_on=[], shape="box", color="lightgreen"
            ),
        )
    )
    return model_ops
