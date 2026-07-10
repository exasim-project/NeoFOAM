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

The only C++ that remains is the mixture *model*
(``multiphase.immiscibleIncompressibleTwoPhaseMixture``). Sub-cycling
(``nAlphaSubCycles > 1``) would need OpenFOAM's ``subCycle`` machinery and is
not supported by this pure-Python path (it raises ``NotImplementedError``).

Provides the shared VoF fields (via ``shared_field_build_steps``) and the
``alpha_advection`` operation (a Python-orchestrated MULES corrector loop).
"""

from typing import Annotated

import pybFoam as pyf
import pybFoam.fvm as fvm
from pybFoam import (
    Info,
    fvc,
    mules as mules_lib,
    surfaceScalarField,
    volScalarField,
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
from neofoam.framework.operations import (
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata

from ..advectionModel import Model, advectionModel
from ..shared import MixtureProtocol, shared_field_build_steps

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


# ---------------------------------------------------------------------------
# Build: the shared VoF fields (phi, mixture, alpha1/alpha2, rho, rhoPhi)
# ---------------------------------------------------------------------------


@mules.build
def build(self: object) -> list[object]:
    """Register the field/model initialisation steps for MULES advection."""
    return shared_field_build_steps()


# ---------------------------------------------------------------------------
# Composable Python steps — each is a small transcription of one block of
# alphaEqn.H over generic pybFoam operators.  Swap / recompose freely.
# ---------------------------------------------------------------------------


def read_alpha_controls(alpha_name: str) -> tuple[int, int, bool]:
    """Read (nAlphaCorr, nAlphaSubCycles, MULESCorr) from ``system/fvSolution``.

    Re-read on every alpha solve — faithful to alphaEqn.H, which pulls the
    controls from ``mesh.solverDict`` each time step (runTimeModifiable).
    Absent dicts/keys fall back to the interFoam defaults with a logged notice;
    a malformed value is a fatal OpenFOAM IO error (not catchable from Python).
    ``found``/``subDict`` regex-match the OpenFOAM solver keys (damBreak:
    ``"alpha.water.*"``).
    """
    defaults = (1, 1, False)
    try:
        fv_solution = pyf.dictionary.read("system/fvSolution")
    except RuntimeError as err:
        Info(
            f"read_alpha_controls: cannot read system/fvSolution ({err}); using defaults."
        )
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
    )


def interface_compression_velocity(
    mixture: MixtureProtocol, phi: surfaceScalarField
) -> surfaceScalarField:
    """phic = cAlpha * |phi / magSf|, with non-coupled boundary faces zeroed.

    Mirrors the ``phic`` block at the top of ``alphaEqn.H``.  The boundary
    zeroing is done in Python via ``fvPatch.coupled()``.
    """
    mesh = phi.mesh()
    phic = surfaceScalarField(
        pyf.Word("phic"), mixture.cAlpha() * pyf.mag(phi / mesh.magSf())
    )
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
    """
    phir = surfaceScalarField(pyf.Word("phir"), phic * mixture.nHatf())
    neg_phir = surfaceScalarField(pyf.Word("negPhir"), -phir)
    phir_flux = surfaceScalarField(
        pyf.Word("phirFlux"), fvc.flux(neg_phir, alpha2, key=_ALPHAR_SCHEME)
    )
    neg_phir_flux = surfaceScalarField(pyf.Word("negPhirFlux"), -phir_flux)
    return surfaceScalarField(
        pyf.Word("alphaPhiUn"),
        fvc.flux(phi, alpha1, key=_ALPHA_SCHEME)
        + fvc.flux(neg_phir_flux, alpha1, key=_ALPHAR_SCHEME),
    )


def mules_implicit_predictor(
    alpha1: volScalarField, phi: surfaceScalarField
) -> surfaceScalarField:
    """Implicit upwind predictor for the MULESCorr branch.

    Builds and solves ``fvm.ddt(alpha1) + fvm.div(phi, alpha1, scheme="Gauss upwind") = 0``
    (Euler ddt from fvSchemes, upwind convection from the inline scheme spec —
    matching alphaEqn.H's hardcoded upwind), modifies ``alpha1`` in-place, and
    returns the resulting upwind face flux.
    """
    alpha1_eqn = pyf.fvScalarMatrix(
        fvm.ddt(alpha1) + fvm.div(phi, alpha1, scheme="Gauss upwind")
    )
    alpha1_eqn.solve()
    return surfaceScalarField(pyf.Word("alphaPhi10"), alpha1_eqn.flux())


def update_rho_rhophi(
    rho: volScalarField,
    rhoPhi: surfaceScalarField,
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    alpha_phi10: surfaceScalarField,
    mixture: MixtureProtocol,
) -> None:
    """Update ``rhoPhi`` and ``rho`` from the phase flux (Euler scheme, phiCN = phi).

    rhoPhi = alphaPhi10 * (rho1 - rho2) + phi * rho2
    rho    = alpha1 * rho1 + alpha2 * rho2
    """
    rho1 = mixture.rho1()
    rho2 = mixture.rho2()
    rhoPhi.assign(alpha_phi10 * rho1 - alpha_phi10 * rho2 + phi * rho2)
    rho.assign(alpha1 * rho1 + alpha2 * rho2)


def _solve_alpha_python(
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    rhoPhi: surfaceScalarField,
    rho: volScalarField,
    mixture: MixtureProtocol,
) -> None:
    """Pure-Python MULES phase-fraction advection (transcription of alphaEqn.H).

    1. Read solver settings (nAlphaCorr, nAlphaSubCycles, MULESCorr).
    2. ``nAlphaSubCycles == 1``: compose the Python steps below.

    ``nAlphaSubCycles > 1`` is not supported by this pure-Python path (it needs
    OpenFOAM's ``subCycle<volScalarField>`` machinery) and raises
    ``NotImplementedError``.
    """
    n_alpha_corr, n_alpha_sub_cycles, mules_corr = read_alpha_controls(alpha1.name())

    # Sub-cycling needs OpenFOAM's subCycle machinery, which this Python path
    # does not implement.
    if n_alpha_sub_cycles > 1:
        raise NotImplementedError(
            "MULES alpha sub-cycling (nAlphaSubCycles > 1) is not supported by "
            "the pure-Python advection path; set nAlphaSubCycles 1 in fvSolution."
        )

    # (a) Interface compression velocity.
    phic = interface_compression_velocity(mixture, phi)

    # (b) Initialise face alpha-flux accumulator.
    alpha_phi10 = surfaceScalarField(
        pyf.Word("alphaPhi10"), phi * fvc.interpolate(alpha1)
    )

    # (c) MULESCorr implicit-upwind predictor.
    if mules_corr:
        alpha_phi10.assign(mules_implicit_predictor(alpha1, phi))
        alpha2.assign(-alpha1 + 1.0)
        mixture.correct()

    # (d) Corrector loop.
    for a_corr in range(n_alpha_corr):
        alpha_phi_un = alpha_phase_flux(phi, alpha1, alpha2, phic, mixture)

        if mules_corr:
            # Capture alpha1 before this iteration's correction so aCorr > 0 can
            # under-relax (mirrors alpha10 in alphaEqn.H).
            alpha10_iter = volScalarField(pyf.Word("alpha10"), 1.0 * alpha1)

            # Correction flux relative to the upwind base; MULES limits it.
            alpha_phi_corr = surfaceScalarField(
                pyf.Word("alphaPhi1Corr"), alpha_phi_un - alpha_phi10
            )
            mules_lib.correct(alpha1, alpha_phi_un, alpha_phi_corr)

            if a_corr == 0:
                alpha_phi10.assign(alpha_phi10 + alpha_phi_corr)
            else:
                alpha1.assign(0.5 * alpha1 + 0.5 * alpha10_iter)
                alpha_phi10.assign(alpha_phi10 + 0.5 * alpha_phi_corr)
        else:
            alpha_phi10.assign(alpha_phi_un)
            mules_lib.explicit_solve(alpha1, phi, alpha_phi10)

        alpha2.assign(-alpha1 + 1.0)
        mixture.correct()

    # (e) Update rhoPhi and rho.
    update_rho_rhophi(rho, rhoPhi, alpha1, alpha2, phi, alpha_phi10, mixture)

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
    mixture: Annotated[MixtureProtocol, "models"],
) -> FieldUpdates:
    """Solve alpha equation (MULES) and update rho / rhoPhi."""
    _solve_alpha_python(alpha1, alpha2, phi, rhoPhi, rho, mixture)
    return FieldUpdates(
        {"alpha1": alpha1, "alpha2": alpha2, "rho": rho, "rhoPhi": rhoPhi}
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
