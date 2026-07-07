# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""AlphaAdvection model for incompressibleVoF (interFoam-based) solver.

Implements MULES-based phase-fraction transport as a standalone core model.

Most of the algorithm logic is written in Python; the MULES inner loop is
delegated to low-level ``vof`` C++ primitives that mirror ``alphaEqn.H``
piece-by-piece:

* ``vof.compute_interface_compression_velocity`` — phic with BC zeroing
* ``vof.alpha_phase_flux``                       — scheme-based alphaPhiUn
* ``vof.mules_explicit_solve``                   — MULES::explicitSolve
* ``vof.mules_correct``                          — MULES::correct
* ``vof.mules_implicit_predictor``               — implicit upwind predictor

Fields provided:
  - phi       (face volumetric flux)
  - alpha1    (phase-1 volume fraction)
  - alpha2    (phase-2 volume fraction)
  - rho       (mixture density)
  - rhoPhi    (density-weighted face flux)

Models provided:
  - mixture   (immiscibleIncompressibleTwoPhaseMixture)

Operations:
  - alpha_advection  (Python-orchestrated MULES subcycling)
"""

from typing import Annotated, Any, Protocol

import pybFoam as pyf
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
    def rho1(self) -> object: ...
    def rho2(self) -> object: ...
    def correct(self) -> None: ...


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
# Python-level alpha advection (calls MULES via C++ binding)
# ---------------------------------------------------------------------------


def _solve_alpha_python(
    alpha1: volScalarField,
    alpha2: volScalarField,
    phi: surfaceScalarField,
    rhoPhi: surfaceScalarField,
    rho: volScalarField,
    mixture: MixtureProtocol,
) -> None:
    """Python port of alphaEqn.H — MULES phase-fraction advection.

    Structure mirrors ``alphaEqn.H`` from OpenFOAM-v2406:

    1. Read solver settings (nAlphaCorr, nAlphaSubCycles, MULESCorr).
    2. For ``nAlphaSubCycles > 1``: delegate to C++ ``vof.solveAlpha``
       (uses OpenFOAM's ``subCycle<volScalarField>`` which cannot be
       replicated in Python without extra bindings).
    3. For ``nAlphaSubCycles == 1``:
       a. Compute interface compression velocity ``phic`` (C++, handles
          non-coupled boundary zeroing).
       b. Initialise alpha face flux ``alpha_phi10``.
       c. **MULESCorr predictor** (if enabled): implicit upwind solve,
          store upwind flux in ``alpha_phi10``.
       d. **Corrector loop** (``nAlphaCorr`` iterations):
          - Compute scheme-based phase flux ``alpha_phi_un`` (C++).
          - MULESCorr  → ``vof.mules_correct``; correct with under-relaxation
                          for iterations > 0 using a per-iteration alpha snapshot.
          - No MULESCorr → ``vof.mules_explicit_solve``.
          - ``alpha2 = 1 - alpha1``; ``mixture.correct()``.
    4. Update ``rhoPhi`` and ``rho`` from ``alpha_phi10`` (Euler scheme formula).
    """
    alpha_name = alpha1.name()

    # ------------------------------------------------------------------
    # 1. Read solver settings
    # ------------------------------------------------------------------
    alpha_scheme = "div(phi,alpha)"
    alphar_scheme = "div(phirb,alpha)"

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

    rho1 = mixture.rho1()
    rho2 = mixture.rho2()

    # ------------------------------------------------------------------
    # 2. Sub-cycling: delegate entirely to C++ (subCycle machinery)
    # ------------------------------------------------------------------
    if n_alpha_sub_cycles > 1:
        vof.solveAlpha(alpha1, alpha2, phi, rhoPhi, rho, mixture)
        return

    # ------------------------------------------------------------------
    # 3. Single cycle: Python-orchestrated MULES following alphaEqn.H
    # ------------------------------------------------------------------

    # (a) Interface compression velocity phic = cAlpha * |phi/magSf|
    #     with non-coupled boundary faces zeroed (C++ handles forAll loop).
    phic = vof.compute_interface_compression_velocity(mixture, phi)

    # (b) Initialise face alpha-flux accumulator
    alpha_phi10 = surfaceScalarField(
        pyf.Word("alphaPhi10"),
        phi * fvc.interpolate(alpha1),
    )

    # (c) MULESCorr implicit upwind predictor
    if mules_corr:
        # Builds fvScalarMatrix(fvmDdt(alpha1) + fvmDiv(phi_upwind, alpha1)),
        # solves it, modifies alpha1 in-place, returns the upwind face flux.
        upwind_flux = vof.mules_implicit_predictor(alpha1, phi)
        alpha_phi10.assign(upwind_flux)
        alpha2.assign(-alpha1 + 1.0)
        mixture.correct()

    # (d) Corrector loop
    for a_corr in range(n_alpha_corr):
        # Scheme-based phase flux with interface compression:
        #   alphaPhiUn = fvc::flux(phi, alpha1, scheme)
        #              + fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme),
        #                          alpha1, alpharScheme)
        # where phir = phic * mixture.nHatf()
        alpha_phi_un = vof.alpha_phase_flux(
            phi, alpha1, alpha2, phic, mixture, alpha_scheme, alphar_scheme
        )

        if mules_corr:
            # Capture alpha1 state BEFORE this iteration's correction
            # so we can under-relax for aCorr > 0 (mirrors alpha10 in alphaEqn.H).
            alpha10_iter = volScalarField(pyf.Word("alpha10"), 1.0 * alpha1)

            # Correction flux relative to the upwind base
            alpha_phi_corr = surfaceScalarField(
                pyf.Word("alphaPhi1Corr"),
                alpha_phi_un - alpha_phi10,
            )

            # Apply MULES limiter to the correction:
            #   MULES::correct(1, alpha1, alphaPhiUn, alphaPhi1Corr, 0, 0, 1, 0)
            # Modifies alpha1 and alpha_phi_corr in-place.
            vof.mules_correct(alpha1, alpha_phi_un, alpha_phi_corr)

            # Under-relax: first corrector adds full correction;
            # subsequent correctors average new vs. pre-corrected value.
            if a_corr == 0:
                alpha_phi10.assign(alpha_phi10 + alpha_phi_corr)
            else:
                alpha1.assign(0.5 * alpha1 + 0.5 * alpha10_iter)
                alpha_phi10.assign(alpha_phi10 + 0.5 * alpha_phi_corr)
        else:
            # Set alpha_phi10 to the scheme flux, then let MULES::explicitSolve
            # update both alpha1 and alpha_phi10 in-place.
            alpha_phi10.assign(alpha_phi_un)
            vof.mules_explicit_solve(alpha1, phi, alpha_phi10)

        # alpha2 = 1 - alpha1
        alpha2.assign(-alpha1 + 1.0)
        mixture.correct()

    # ------------------------------------------------------------------
    # 4. Update rhoPhi and rho (Euler scheme: phiCN = phi)
    #   rhoPhi = alphaPhi10 * (rho1 - rho2) + phi * rho2
    #   rho    = alpha1 * rho1 + alpha2 * rho2
    # ------------------------------------------------------------------
    rhoPhi.assign(alpha_phi10 * rho1 - alpha_phi10 * rho2 + phi * rho2)  # type: ignore[operator]
    rho.assign(alpha1 * rho1 + alpha2 * rho2)  # type: ignore[operator]

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
