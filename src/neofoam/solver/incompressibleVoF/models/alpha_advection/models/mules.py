# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MULES alpha-advection scheme (interFoam-based).

Implements MULES-based phase-fraction transport as a member of the
``advectionModel`` family. The whole algorithm is written in Python — a
step-by-step transcription of ``alphaEqn.H`` from OpenFOAM-v2406 — composed
from *generic* pybFoam operator primitives (nothing VoF-specific is delegated
to C++):

* interface compression velocity ``phic`` — ``pyf.mag`` + ``fvPatch.coupled()``
* scheme-based phase flux ``alphaPhiUn``  — ``fvc.flux(phi, field, key=...)``
* implicit-upwind predictor              — ``fvm.ddt`` + ``fvm.div(.., scheme="Gauss upwind")``
* MULES limiter                          — ``mules.explicit_solve`` / ``mules.correct``
* ddt off-centring ``ocCoeff``/``phiCN`` — ``fvMesh.ddtSchemeInfo``

The only C++ that remains is the mixture *model*
(``multiphase.immiscibleIncompressibleTwoPhaseMixture``). The primitives and
the corrector loop live in :mod:`neofoam.algorithms.mules`, shared with the
advection-only VoF backends; this module adds the mixture tails. Sub-cycling
(``nAlphaSubCycles > 1``) is Python too — ``sub_cycled_alpha_eqn`` reproduces
``subCycle<volScalarField>`` on ``Foam::Time`` itself.

Provides the shared VoF fields (via ``shared_field_build_steps``) and the
``alpha_advection`` operation (a Python-orchestrated MULES corrector loop).
"""

from typing import Annotated, Optional

import pybFoam as pyf
from pybFoam import (
    Info,
    dimensionedScalar,
    dimless,
    surfaceScalarField,
    volScalarField,
)

import neofoam.algorithms.mules as shared_mules
from neofoam.algorithms.mules import (
    AlphaControls,
    AlphaFluxes,
    InterfaceProtocol,
    OpenFOAMMulesOps,
    PhaseFractions,
    VolumetricFlux,
    crank_nicolson_flux,
    read_alpha_controls,
    read_alpha_ddt_off_centring,
)
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
from neofoam.framework.initialization import model
from neofoam.framework.operations import (
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..advectionModel import Model, advectionModel
from ..shared import (
    MixtureProtocol,
    shared_field_build_steps,
)

__all__ = ["mules"]

mules = Model("MULES").register_with(advectionModel).labeled("MULES")

# Per-spec fvSchemes / fvSolution slices + the ``0/alpha.water`` field the wizard
# authors. Schema-only (surfaced through ``configurations(solver)``); the alpha
# field is created at run time from the C++ mixture (``shared_field_build_steps``),
# so declaring it here changes no init behaviour. ``alpha.water`` is declared on
# MULES alone (the default advection scheme) so it surfaces exactly once even
# though every advection member owns the same field.
MulesFvSchemes = mules.config(fvSchemes)
MulesFvSolution = mules.config(fvSolution)

mules.field(
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


# The momentum ddt scheme alphaEqn.H's rhoPhi expression branches on; falls back to
# the dict's ``default``.
_MOMENTUM_DDT = "ddt(rho,U)"
_EULER_DDT_SCHEMES = ("Euler", "localEuler")


# ---------------------------------------------------------------------------
# Build: the shared VoF fields (phi, mixture, alpha1/alpha2, rho, rhoPhi)
# ---------------------------------------------------------------------------


@mules.build
def build(self: object) -> list[object]:
    """Register the field/model initialisation steps for MULES advection.

    On top of the shared VoF fields, the ``talphaPhi1Corr0`` slot of
    ``createAlphaFluxes.H``: one mutable holder for the whole run.
    """
    return [*shared_field_build_steps(), model("alphaPhi1Corr0", lambda _: [None])]


# ---------------------------------------------------------------------------
# Composable Python steps — the alphaEqn.H loop itself lives in
# neofoam.algorithms.mules; these are the mixture-specific tails.
# ---------------------------------------------------------------------------


def update_rho_phi(
    rhoPhi: surfaceScalarField,
    phi: surfaceScalarField,
    phi_cn: surfaceScalarField,
    alpha_phi10: surfaceScalarField,
    mixture: MixtureProtocol,
    oc_coeff: float,
) -> None:
    """``rhoPhi`` from the phase flux — the tail of ``alphaEqn.H``.

    Runs once per alpha pass, including once per sub-cycle. The *momentum* ddt
    scheme picks the expression: Euler ``ddt(rho,U)`` wants the flux the alpha
    equation just used, anything else the end-of-time-step one.
    """
    rho1 = mixture.rho1()
    rho2 = mixture.rho2()
    scheme_name, _ = phi.mesh().ddtSchemeInfo(pyf.Word(_MOMENTUM_DDT))
    if scheme_name in _EULER_DDT_SCHEMES:
        rhoPhi.assign(alpha_phi10 * rho1 - alpha_phi10 * rho2 + phi_cn * rho2)
        return

    if oc_coeff > 0:
        cn_coeff = 1.0 / (1.0 + oc_coeff)
        alpha_phi10.assign(
            (alpha_phi10 - (1.0 - cn_coeff) * alpha_phi10.oldTime())
            / dimensionedScalar(pyf.Word("cnCoeff"), dimless, cn_coeff)
        )
    rhoPhi.assign(alpha_phi10 * rho1 - alpha_phi10 * rho2 + phi * rho2)


def update_rho(
    rho: volScalarField,
    alpha1: volScalarField,
    alpha2: volScalarField,
    mixture: MixtureProtocol,
) -> None:
    """``rho = alpha1*rho1 + alpha2*rho2`` — the last line of ``alphaEqnSubCycle.H``.

    Once per time step, *after* any sub-cycle: writing ``rho`` rolls its old-time
    value, and the momentum ``fvm::ddt(rho, U)`` needs the one from the start of
    the real time step, not from a sub-step.
    """
    rho.assign(alpha1 * mixture.rho1() + alpha2 * mixture.rho2())


def alpha_eqn(
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    phi_cn: surfaceScalarField,
    mixture: InterfaceProtocol,
    n_alpha_corr: int,
    mules_corr: bool,
    alpha_apply_prev_corr: bool,
    alphaPhiUn: surfaceScalarField,
    alphaPhi10: surfaceScalarField,
    alphaPhi1Corr0: list[Optional[surfaceScalarField]],
) -> surfaceScalarField:
    """One pass of ``alphaEqn.H``: advance ``alpha1``/``alpha2``, return alphaPhi10.

    The pybFoam entry to :func:`neofoam.algorithms.mules.alpha_eqn`.
    ``alphaPhi1Corr0`` is the ``talphaPhi1Corr0`` holder that survives between
    passes.
    """
    fluxes = AlphaFluxes(alphaPhiUn, alphaPhi10, alphaPhi1Corr0[0])
    alpha_phi10: surfaceScalarField = shared_mules.alpha_eqn(
        OpenFOAMMulesOps(alpha1, mixture),
        PhaseFractions(alpha1, alpha2),
        VolumetricFlux(phi, phi_cn),
        AlphaControls(n_alpha_corr, 1, mules_corr, alpha_apply_prev_corr),
        fluxes,
    )
    alphaPhi1Corr0[0] = fluxes.corr0
    return alpha_phi10


def sub_cycled_alpha_eqn(
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    rhoPhi: surfaceScalarField,
    rho: volScalarField,
    mixture: MixtureProtocol,
    n_alpha_corr: int,
    mules_corr: bool,
    alpha_apply_prev_corr: bool,
    n_alpha_sub_cycles: int,
    alphaPhiUn: surfaceScalarField,
    alphaPhi10: surfaceScalarField,
    alphaPhi1Corr0: list[Optional[surfaceScalarField]],
) -> None:
    """Transcription of ``alphaEqnSubCycle.H``: n alpha passes per time step.

    Each pass advances ``alpha1`` by ``deltaT/n`` with the *full* flux ``phi``,
    driven by a real ``Foam::Time`` sub-cycle
    (:func:`neofoam.algorithms.mules.sub_cycled_alpha_eqn`). Only ``rhoPhi``
    accumulates across sub-steps.
    """
    rho_phi_sum = surfaceScalarField(pyf.Word("rhoPhiSum"), 0.0 * rhoPhi)
    fluxes = AlphaFluxes(alphaPhiUn, alphaPhi10, alphaPhi1Corr0[0])
    passes = shared_mules.sub_cycled_alpha_eqn(
        OpenFOAMMulesOps(alpha1, mixture),
        PhaseFractions(alpha1, alpha2),
        VolumetricFlux(phi, phi),
        AlphaControls(n_alpha_corr, n_alpha_sub_cycles, mules_corr, alpha_apply_prev_corr),
        fluxes,
    )
    for fraction, alpha_phi10 in passes:
        update_rho_phi(rhoPhi, phi, phi, alpha_phi10, mixture, 0.0)
        rho_phi_sum.assign(rho_phi_sum + fraction * rhoPhi)
    alphaPhi1Corr0[0] = fluxes.corr0

    rhoPhi.assign(rho_phi_sum)
    update_rho(rho, alpha1, alpha2, mixture)


def _solve_alpha_python(
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    rhoPhi: surfaceScalarField,
    rho: volScalarField,
    mixture: MixtureProtocol,
    alphaPhiUn: surfaceScalarField,
    alphaPhi10: surfaceScalarField,
    alphaPhi1Corr0: list[Optional[surfaceScalarField]],
) -> None:
    """Pure-Python MULES phase-fraction advection (transcription of alphaEqn.H)."""
    n_alpha_corr, n_alpha_sub_cycles, mules_corr, alpha_apply_prev_corr = read_alpha_controls(
        alpha1.name()
    )
    oc_coeff = read_alpha_ddt_off_centring(alpha1.mesh(), n_alpha_sub_cycles)

    if n_alpha_sub_cycles > 1:
        sub_cycled_alpha_eqn(
            alpha1,
            alpha2,
            phi,
            rhoPhi,
            rho,
            mixture,
            n_alpha_corr,
            mules_corr,
            alpha_apply_prev_corr,
            n_alpha_sub_cycles,
            alphaPhiUn,
            alphaPhi10,
            alphaPhi1Corr0,
        )
    else:
        phi_cn = crank_nicolson_flux(phi, oc_coeff)
        alpha_phi10 = alpha_eqn(
            alpha1,
            alpha2,
            phi,
            phi_cn,
            mixture,
            n_alpha_corr,
            mules_corr,
            alpha_apply_prev_corr,
            alphaPhiUn,
            alphaPhi10,
            alphaPhi1Corr0,
        )
        update_rho_phi(rhoPhi, phi, phi_cn, alpha_phi10, mixture, oc_coeff)
        update_rho(rho, alpha1, alpha2, mixture)

    # interFoam.C:154 — one more correct() after the whole alphaEqnSubCycle.H
    # block; an alphaContactAngle patch rewrites alpha1's gradient on every call.
    mixture.correct()

    Info(f"Phase-1 volume fraction: nAlphaCorr={n_alpha_corr}  MULESCorr={mules_corr}")


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------


@mules.operation(operation_number="2.0")
@MulesFvSchemes.add(
    ddt="ddt(alpha1)",
    # scheme-based phase flux + interface-compression flux (alphaEqn.H).
    div=["div(phi,alpha)", "div(phirb,alpha)"],
)
@MulesFvSolution.add("alpha.water")
def alpha_advection(
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    rhoPhi: surfaceScalarField,
    rho: volScalarField,
    alphaPhiUn: surfaceScalarField,
    alphaPhi10: surfaceScalarField,
    mixture: Annotated[MixtureProtocol, "models"],
    alphaPhi1Corr0: Annotated[list[Optional[surfaceScalarField]], "models"],
) -> FieldUpdates:
    """Solve alpha equation (MULES) and update rho / rhoPhi / alphaPhiUn / alphaPhi10."""
    _solve_alpha_python(
        alpha1, alpha2, phi, rhoPhi, rho, mixture, alphaPhiUn, alphaPhi10, alphaPhi1Corr0
    )
    return FieldUpdates(
        {
            "alpha1": alpha1,
            "alpha2": alpha2,
            "rho": rho,
            "rhoPhi": rhoPhi,
            "alphaPhiUn": alphaPhiUn,
            "alphaPhi10": alphaPhi10,
        }
    )


# ---------------------------------------------------------------------------
# Operation collection: expose alpha_advection
# ---------------------------------------------------------------------------


@mules.operation_collection
def collected_operations(self: object) -> Operations:
    # The collection path bypasses the spec's default operation wrapping, so
    # wrap alpha_advection with dependency resolution here (``self`` is the
    # bound runtime).
    wrapped_alpha_advection = mules.wrap_operation(alpha_advection, self)
    model_ops = Operations()
    model_ops.add(
        Operation(
            func=SequentialOp(wrapped_alpha_advection),
            metadata=OperationMetadata(
                op_name="alpha_advection",
                depends_on=[],
                before=[],
                shape="box",
                color="lightgreen",
            ),
        )
    )
    return model_ops
