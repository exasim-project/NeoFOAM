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
(``multiphase.immiscibleIncompressibleTwoPhaseMixture``). Sub-cycling
(``nAlphaSubCycles > 1``) is Python too — ``sub_cycled_alpha_eqn`` reproduces
``subCycle<volScalarField>`` on ``Foam::Time`` itself.

Provides the shared VoF fields (via ``shared_field_build_steps``) and the
``alpha_advection`` operation (a Python-orchestrated MULES corrector loop).
"""

from typing import Annotated, Optional

import pybFoam as pyf
import pybFoam.fvm as fvm
from pybFoam import (
    Info,
    dimensionedScalar,
    dimless,
    fvc,
    surfaceScalarField,
    volScalarField,
)
from pybFoam import (
    mules as mules_lib,
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
from ..shared import MixtureProtocol, alpha_sub_cycle, shared_field_build_steps

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


# Scheme names, resolved from fvSchemes divSchemes (mirrors alphaEqn.H).
_ALPHA_SCHEME = "div(phi,alpha)"
_ALPHAR_SCHEME = "div(phirb,alpha)"

# fvSchemes ddtSchemes entries alphaEqn.H branches on: the alpha equation's own
# ddt (which sets the off-centring) and the momentum ddt (which picks the rhoPhi
# expression). Both fall back to the dict's ``default``.
_ALPHA_DDT = "ddt(alpha)"
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
# Composable Python steps — each is a small transcription of one block of
# alphaEqn.H over generic pybFoam operators.  Swap / recompose freely.
# ---------------------------------------------------------------------------


def read_alpha_controls(alpha_name: str) -> tuple[int, int, bool, bool]:
    """Read (nAlphaCorr, nAlphaSubCycles, MULESCorr, alphaApplyPrevCorr) from
    ``system/fvSolution``.

    Re-read on every alpha solve — faithful to alphaEqn.H, which pulls the
    controls from ``mesh.solverDict`` each time step (runTimeModifiable).
    Absent dicts/keys fall back to the interFoam defaults with a logged notice;
    a malformed value is a fatal OpenFOAM IO error (not catchable from Python).
    ``found``/``subDict`` regex-match the OpenFOAM solver keys (damBreak:
    ``"alpha.water.*"``).
    """
    defaults = (1, 1, False, False)
    try:
        fv_solution = pyf.dictionary.read("system/fvSolution")
    except RuntimeError as err:
        Info(f"read_alpha_controls: cannot read system/fvSolution ({err}); using defaults.")
        return defaults
    if not fv_solution.found("solvers"):
        Info("read_alpha_controls: no 'solvers' dict in fvSolution; using defaults.")
        return defaults
    solvers = fv_solution.subDict("solvers")
    if not solvers.found(alpha_name):
        Info(f"read_alpha_controls: no '{alpha_name}' solver dict; using defaults.")
        return defaults
    alpha_dict = solvers.subDict(alpha_name)
    return (
        alpha_dict.getOrDefault[int]("nAlphaCorr", 1),
        alpha_dict.getOrDefault[int]("nAlphaSubCycles", 1),
        alpha_dict.getOrDefault[bool]("MULESCorr", False),
        alpha_dict.getOrDefault[bool]("alphaApplyPrevCorr", False),
    )


def alpha_ddt_off_centring(
    scheme_name: str,
    scheme_coefficient: float,
    n_alpha_sub_cycles: int,
    after_first_time_step: bool,
) -> float:
    """``ocCoeff`` of ``alphaEqn.H`` for an already-read ``ddt(alpha)`` scheme.

    The decision table on its own, so it can be exercised without a mesh; use
    ``read_alpha_ddt_off_centring`` to get the inputs from a live case. 0 means
    "integrate exactly as Euler". Raises for the configurations native rejects.

    >>> alpha_ddt_off_centring("CrankNicolson", 0.5, 1, True)
    0.5
    """
    if scheme_name in _EULER_DDT_SCHEMES:
        return 0.0
    if scheme_name != "CrankNicolson":
        raise ValueError(
            f"ddtSchemes/{_ALPHA_DDT} is '{scheme_name}': only Euler and "
            "CrankNicolson ddt schemes are supported for the alpha equation."
        )
    if n_alpha_sub_cycles > 1:
        raise ValueError(
            "Sub-cycling is not supported with the CrankNicolson ddt scheme "
            f"(nAlphaSubCycles = {n_alpha_sub_cycles})."
        )
    # The scheme needs an old-time alpha flux it does not have yet, so the first
    # step of a fresh run is integrated with Euler.
    return scheme_coefficient if after_first_time_step else 0.0


def read_alpha_ddt_off_centring(mesh: pyf.fvMesh, n_alpha_sub_cycles: int) -> float:
    """``ocCoeff`` for the alpha equation, read from the case's ``system/fvSchemes``.

    Re-read on every alpha solve, as alphaEqn.H does: the ``CrankNicolson``
    coefficient may be a ``Function1`` of time. A restart is treated as a fresh
    start — ``alphaPhi0.<phase>`` is neither written nor read here.
    """
    scheme_name, scheme_coefficient = mesh.ddtSchemeInfo(pyf.Word(_ALPHA_DDT))
    runtime = mesh.time()
    return alpha_ddt_off_centring(
        scheme_name,
        scheme_coefficient,
        n_alpha_sub_cycles,
        runtime.timeIndex() > runtime.startTimeIndex() + 1,
    )


def crank_nicolson_flux(phi: surfaceScalarField, oc_coeff: float) -> surfaceScalarField:
    """The off-centred volumetric flux ``phiCN`` the alpha equation advects with.

    At ``ocCoeff == 0`` ``phi`` is returned unchanged, so the Euler path never
    reads ``phi``'s old-time slot — doing so would roll it where native does not.
    """
    if oc_coeff <= 0:
        return phi
    cn_coeff = 1.0 / (1.0 + oc_coeff)
    return surfaceScalarField(pyf.Word("phiCN"), cn_coeff * phi + (1.0 - cn_coeff) * phi.oldTime())


def interface_compression_velocity(
    mixture: MixtureProtocol, phi: surfaceScalarField
) -> surfaceScalarField:
    """phic = cAlpha * |phi / magSf|, with non-coupled boundary faces zeroed.

    Mirrors the ``phic`` block at the top of ``alphaEqn.H``.  The boundary
    zeroing is done in Python via ``fvPatch.coupled()``.
    """
    mesh = phi.mesh()
    phic = surfaceScalarField(pyf.Word("phic"), mixture.cAlpha() * pyf.mag(phi / mesh.magSf()))
    # Zero compression on non-coupled (inlet/outlet/wall) boundary faces.
    # Index access (fvPatch by reference) — the fvBoundaryMesh iterator returns
    # fvPatch by value, which is invalid for the abstract fvPatch base.
    boundary = mesh.boundary()
    for i in range(len(boundary)):
        patch = boundary[i]
        if not patch.coupled():
            name = str(patch.name())
            phic[name] = pyf.scalarField([0.0] * len(phic[name]))
    return phic


def alpha_phase_flux(
    phi: surfaceScalarField,
    alpha1: volScalarField,
    alpha2: volScalarField,
    phic: surfaceScalarField,
    mixture: MixtureProtocol,
    alphaPhiUn: surfaceScalarField,
) -> surfaceScalarField:
    """Scheme-based phase flux with interface compression (``alphaPhiUn``).

    alphaPhiUn = fvc::flux(phi, alpha1, alphaScheme)
               + fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme),
                            alpha1, alpharScheme)
    with phir = phic * mixture.nHatf().

    Composed from ``fvc.flux(phi, field, key=...)`` (by-name fvSchemes lookup).
    The negated intermediate fluxes are materialised into ``surfaceScalarField``
    (bitwise-identical) so they satisfy the ``fvc.flux`` surface-field
    first-argument overload.

    Writes into the persistent ``alphaPhiUn`` (registered once by
    createAlphaFluxes.H) via ``.assign``: a same-named local would deregister it
    on return.
    """
    phir = surfaceScalarField(pyf.Word("phir"), phic * mixture.nHatf())
    neg_phir = surfaceScalarField(pyf.Word("negPhir"), -phir)
    phir_flux = surfaceScalarField(
        pyf.Word("phirFlux"), fvc.flux(neg_phir, alpha2, key=_ALPHAR_SCHEME)
    )
    neg_phir_flux = surfaceScalarField(pyf.Word("negPhirFlux"), -phir_flux)
    alphaPhiUn.assign(
        fvc.flux(phi, alpha1, key=_ALPHA_SCHEME)
        + fvc.flux(neg_phir_flux, alpha1, key=_ALPHAR_SCHEME)
    )
    return alphaPhiUn


def mules_implicit_predictor(
    alpha1: volScalarField, phi_cn: surfaceScalarField
) -> surfaceScalarField:
    """Implicit upwind predictor for the MULESCorr branch.

    Modifies ``alpha1`` in-place and returns the upwind face flux
    (``talphaPhi1UD``). Both schemes are inline specs rather than fvSchemes
    lookups because alphaEqn.H hardcodes them: the predictor stays Euler even
    under a CrankNicolson ``ddt(alpha)`` (``phiCN`` carries the off-centring).
    """
    alpha1_eqn = pyf.fvScalarMatrix(
        fvm.ddt(alpha1, scheme="Euler") + fvm.div(phi_cn, alpha1, scheme="Gauss upwind")
    )
    alpha1_eqn.solve()
    return surfaceScalarField(pyf.Word("alphaPhi1UD"), alpha1_eqn.flux())


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
    mixture: MixtureProtocol,
    n_alpha_corr: int,
    mules_corr: bool,
    alpha_apply_prev_corr: bool,
    alphaPhiUn: surfaceScalarField,
    alphaPhi10: surfaceScalarField,
    alphaPhi1Corr0: list[Optional[surfaceScalarField]],
) -> surfaceScalarField:
    """One pass of ``alphaEqn.H``: advance ``alpha1``/``alpha2``, return alphaPhi10.

    Native transports with the off-centred ``phi_cn`` and compresses with the raw
    ``phi``; the two must not be swapped. ``alphaPhiUn``/``alphaPhi10`` are the
    persistent flux fields, overwritten in place each corrector — never
    accumulated. ``alphaPhi1Corr0`` is the ``talphaPhi1Corr0`` holder that
    survives between passes.
    """
    # (a) Interface compression velocity.
    phic = interface_compression_velocity(mixture, phi)

    # (b) MULESCorr implicit-upwind predictor.
    if mules_corr:
        # The pure upwind flux stays alive alongside alphaPhi10, which is about to
        # grow by the previous pass's correction; the cache below measures against it.
        alpha_phi1_ud = mules_implicit_predictor(alpha1, phi_cn)
        alphaPhi10.assign(alpha_phi1_ud)

        previous_corr = alphaPhi1Corr0[0]
        if alpha_apply_prev_corr and previous_corr is not None:
            Info("Applying the previous iteration compression flux")
            mules_lib.correct(alpha1, alphaPhi10, previous_corr)
            alphaPhi10.assign(alphaPhi10 + previous_corr)

        alphaPhi1Corr0[0] = alpha_phi1_ud
        alpha2.assign(-alpha1 + 1.0)
        mixture.correct()

    # (c) Corrector loop.
    for a_corr in range(n_alpha_corr):
        alpha_phi_un = alpha_phase_flux(phi, alpha1, alpha2, phic, mixture, alphaPhiUn)

        if mules_corr:
            # Capture alpha1 before this iteration's correction so aCorr > 0 can
            # under-relax (mirrors alpha10 in alphaEqn.H).
            alpha10_iter = volScalarField(pyf.Word("alpha10"), 1.0 * alpha1)

            # Correction flux relative to the upwind base; MULES limits it.
            alpha_phi_corr = surfaceScalarField(
                pyf.Word("alphaPhi1Corr"), alpha_phi_un - alphaPhi10
            )
            mules_lib.correct(alpha1, alpha_phi_un, alpha_phi_corr)

            if a_corr == 0:
                alphaPhi10.assign(alphaPhi10 + alpha_phi_corr)
            else:
                alpha1.assign(0.5 * alpha1 + 0.5 * alpha10_iter)
                alphaPhi10.assign(alphaPhi10 + 0.5 * alpha_phi_corr)
        else:
            alphaPhi10.assign(alpha_phi_un)
            mules_lib.explicit_solve(alpha1, phi_cn, alphaPhi10)

        alpha2.assign(-alpha1 + 1.0)
        mixture.correct()

    # (d) Turn the cached upwind flux into this pass's compression correction, which
    # seeds the next pass. Any other configuration empties the slot, so switching
    # alphaApplyPrevCorr off mid-run cannot leave a stale correction in use.
    cached_upwind = alphaPhi1Corr0[0]
    if mules_corr and alpha_apply_prev_corr and cached_upwind is not None:
        cached_upwind.assign(alphaPhi10 - cached_upwind)
    else:
        alphaPhi1Corr0[0] = None

    return alphaPhi10


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
    driven by a real ``Foam::Time`` sub-cycle. Only ``rhoPhi`` accumulates across
    sub-steps; the flux fields and the ``alphaPhi1Corr0`` cache are re-assigned
    every pass, as native's per-sub-step ``#include`` of alphaEqn.H does.
    Sub-cycling only runs under an Euler ``ddt(alpha)``, so ``phi`` is its own
    off-centred flux.
    """
    total_delta_t = alpha1.mesh().time().deltaTValue()
    rho_phi_sum = surfaceScalarField(pyf.Word("rhoPhiSum"), 0.0 * rhoPhi)

    with alpha_sub_cycle(alpha1, n_alpha_sub_cycles) as runtime:
        for _ in range(n_alpha_sub_cycles):
            runtime.increment()
            alpha_phi10 = alpha_eqn(
                alpha1,
                alpha2,
                phi,
                phi,
                mixture,
                n_alpha_corr,
                mules_corr,
                alpha_apply_prev_corr,
                alphaPhiUn,
                alphaPhi10,
                alphaPhi1Corr0,
            )
            update_rho_phi(rhoPhi, phi, phi, alpha_phi10, mixture, 0.0)
            rho_phi_sum.assign(rho_phi_sum + (runtime.deltaTValue() / total_delta_t) * rhoPhi)

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
