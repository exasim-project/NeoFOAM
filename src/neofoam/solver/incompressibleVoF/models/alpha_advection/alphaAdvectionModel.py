# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""AlphaAdvection model for incompressibleVoF (interFoam-based) solver.

Implements MULES-based phase-fraction transport as a standalone core model.

The whole algorithm is written in Python — a step-by-step transcription of
``alphaEqn.H`` from OpenFOAM-v2406 — composed from *generic* pybFoam operator
primitives (nothing VoF-specific is delegated to C++):

* interface compression velocity ``phic`` — ``pyf.mag`` + ``fvPatch.coupled()``
* scheme-based phase flux ``alphaPhiUn``  — ``fvc.flux(phi, field, key=...)``
* implicit-upwind predictor              — ``fvm.ddt`` + ``fvm.div(.., scheme="Gauss upwind")``
* MULES limiter                          — ``mules.explicit_solve`` / ``mules.correct``

The only C++ that remains is the mixture *model*
(``vof.immiscibleIncompressibleTwoPhaseMixture``) and the ``nAlphaSubCycles > 1``
fallback (``vof.solveAlpha``), which needs OpenFOAM's ``subCycle`` machinery.

Because each step is a small Python function over generic operators, schemes,
correctors and the predictor can be recomposed or replaced without touching C++.

Fields provided:
  - phi       (face volumetric flux)
  - alpha1    (phase-1 volume fraction)
  - alpha2    (phase-2 volume fraction)
  - rho       (mixture density)
  - rhoPhi    (density-weighted face flux)

Models provided:
  - mixture   (immiscibleIncompressibleTwoPhaseMixture)

Operations:
  - alpha_advection  (Python-orchestrated MULES corrector loop)
"""

from typing import Annotated, Any, Protocol

import pybFoam as pyf
import pybFoam.fvm as fvm
import pybFoam.mules as mules
import pybFoam.vof as vof
from pybFoam import (
    Info,
    fvc,
    surfaceScalarField,
    volScalarField,
)

from neofoam.framework.context import FieldUpdates
from neofoam.framework.dependency_resolver import wrap_with_dependency_resolution
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata
from ..incompressibleVoFModel import Model

alpha_advection_model = Model("AlphaAdvection")


# ---------------------------------------------------------------------------
# Type protocol for static analysis
# ---------------------------------------------------------------------------


class MixtureProtocol(Protocol):
    def alpha1(self) -> volScalarField: ...
    def alpha2(self) -> volScalarField: ...
    def rho1(self) -> Any: ...
    def rho2(self) -> Any: ...
    def cAlpha(self) -> float: ...
    def nHatf(self) -> surfaceScalarField: ...
    def correct(self) -> None: ...


# Scheme names, resolved from fvSchemes divSchemes (mirrors alphaEqn.H).
_ALPHA_SCHEME = "div(phi,alpha)"
_ALPHAR_SCHEME = "div(phirb,alpha)"


# ---------------------------------------------------------------------------
# Build: register initialisation steps provided by this model
# ---------------------------------------------------------------------------


@alpha_advection_model.build
def build(self: object) -> list[object]:
    """Register field/model initialisation steps for alpha advection."""

    def create_phi(context: dict[str, Any]) -> surfaceScalarField:
        """Create face flux phi from U."""
        return pyf.createPhi(context["fields.U"])

    def create_mixture(context: dict[str, Any]) -> Any:
        """Create immiscibleIncompressibleTwoPhaseMixture and register alpha flux."""
        mesh = context["mesh"]
        U = context["fields.U"]
        phi = context["fields.phi"]
        mixture = vof.immiscibleIncompressibleTwoPhaseMixture(U, phi)
        # alpha.water face flux required for MULES solver
        mesh.setFluxRequired(mixture.alpha1().name())
        return mixture

    def create_alpha1(context: dict[str, Any]) -> volScalarField:
        return context["models.mixture"].alpha1()  # type: ignore[no-any-return]

    def create_alpha2(context: dict[str, Any]) -> volScalarField:
        return context["models.mixture"].alpha2()  # type: ignore[no-any-return]

    def create_rho(context: dict[str, Any]) -> volScalarField:
        mixture = context["models.mixture"]
        alpha1 = context["fields.alpha1"]
        alpha2 = context["fields.alpha2"]
        rho1 = mixture.rho1()
        rho2 = mixture.rho2()
        rho = volScalarField(pyf.Word("rho"), alpha1 * rho1 + alpha2 * rho2)
        # Store old-time so that fvm.ddt(rho, U) in momentum has a valid old value
        rho.oldTime()  # type: ignore[attr-defined]
        return rho

    def create_rho_phi(context: dict[str, Any]) -> surfaceScalarField:
        rho = context["fields.rho"]
        phi = context["fields.phi"]
        return surfaceScalarField(pyf.Word("rhoPhi"), fvc.interpolate(rho) * phi)

    return [
        field("phi", create_phi, depends_on=["fields.U"]),
        model("mixture", create_mixture, depends_on=["fields.U", "fields.phi", "mesh"]),
        field("alpha1", create_alpha1, depends_on=["models.mixture"]),
        field("alpha2", create_alpha2, depends_on=["models.mixture"]),
        field(
            "rho",
            create_rho,
            depends_on=["models.mixture", "fields.alpha1", "fields.alpha2"],
        ),
        field("rhoPhi", create_rho_phi, depends_on=["fields.rho", "fields.phi"]),
    ]


# ---------------------------------------------------------------------------
# Composable Python steps — each is a small transcription of one block of
# alphaEqn.H over generic pybFoam operators.  Swap / recompose freely.
# ---------------------------------------------------------------------------


def read_alpha_controls(alpha_name: str) -> tuple[int, int, bool]:
    """Read (nAlphaCorr, nAlphaSubCycles, MULESCorr) from ``system/fvSolution``."""
    n_alpha_corr = 1
    n_alpha_sub_cycles = 1
    mules_corr = False
    try:
        fv_solution = pyf.dictionary.read("system/fvSolution")
        alpha_dict = fv_solution.subDict("solvers").subDict(alpha_name)
        n_alpha_corr = alpha_dict.getOrDefault[int]("nAlphaCorr", 1)
        n_alpha_sub_cycles = alpha_dict.getOrDefault[int]("nAlphaSubCycles", 1)
        mules_corr = alpha_dict.getOrDefault[bool]("MULESCorr", False)
    except Exception:
        pass  # use defaults
    return n_alpha_corr, n_alpha_sub_cycles, mules_corr


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
    2. ``nAlphaSubCycles > 1``: delegate to ``vof.solveAlpha`` (needs OpenFOAM's
       ``subCycle<volScalarField>`` machinery).
    3. ``nAlphaSubCycles == 1``: compose the Python steps below.
    """
    n_alpha_corr, n_alpha_sub_cycles, mules_corr = read_alpha_controls(alpha1.name())

    # Sub-cycling still needs OpenFOAM's subCycle machinery.
    if n_alpha_sub_cycles > 1:
        vof.solveAlpha(alpha1, alpha2, phi, rhoPhi, rho, mixture)
        return

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
            mules.correct(alpha1, alpha_phi_un, alpha_phi_corr)

            if a_corr == 0:
                alpha_phi10.assign(alpha_phi10 + alpha_phi_corr)
            else:
                alpha1.assign(0.5 * alpha1 + 0.5 * alpha10_iter)
                alpha_phi10.assign(alpha_phi10 + 0.5 * alpha_phi_corr)
        else:
            alpha_phi10.assign(alpha_phi_un)
            mules.explicit_solve(alpha1, phi, alpha_phi10)

        alpha2.assign(-alpha1 + 1.0)
        mixture.correct()

    # (e) Update rhoPhi and rho.
    update_rho_rhophi(rho, rhoPhi, alpha1, alpha2, phi, alpha_phi10, mixture)

    Info(f"Phase-1 volume fraction: nAlphaCorr={n_alpha_corr}  MULESCorr={mules_corr}")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@alpha_advection_model.operation(operation_number="2.0")
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


@alpha_advection_model.operation_collection
def collected_operations(self: object) -> Operations:
    # The collection path bypasses the spec's default operation wrapping, so
    # wrap alpha_advection with dependency resolution here (``self`` is the
    # bound runtime).
    wrapped_alpha_advection = wrap_with_dependency_resolution(
        alpha_advection, self, alpha_advection_model._dependency_resolver
    )
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
