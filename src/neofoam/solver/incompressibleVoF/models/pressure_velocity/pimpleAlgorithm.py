# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""PIMPLE algorithm for incompressibleVoF (interFoam-based) solver.

Implements:
  - momentum (density-weighted momentum with surface tension)
  - continuity (pressure-velocity coupling with gravity/surface tension)

Note: alpha_advection has been extracted into the AlphaAdvection core model
(see models/alpha_advection/alphaAdvectionModel.py).  This model owns fields:
  phi, mixture, alpha1, alpha2, rho, rhoPhi.
"""

from typing import Annotated, Any, Callable, Protocol

import pybFoam as pyf
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam.algorithms.solution_loop.control import PimpleControl
from neofoam.foam.initialization import read_vol_field
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.dependency_resolver import wrap_with_dependency_resolution
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
from neofoam.framework.types import OperationMetadata
from .control_factory import create_pimple_control
from ..incompressibleVoFModel import Model

pimple = Model("Pimple")


# ---------------------------------------------------------------------------
# Type protocols for static analysis
# ---------------------------------------------------------------------------


class MixtureProtocol(Protocol):
    def alpha1(self) -> volScalarField: ...
    def alpha2(self) -> volScalarField: ...
    def rho1(self) -> object: ...
    def rho2(self) -> object: ...
    def surfaceTensionForce(self) -> surfaceScalarField: ...
    def correct(self) -> None: ...


class TwoPhaseTransportProtocol(Protocol):
    def divDevRhoReff(
        self, rho: volScalarField, U: volVectorField
    ) -> fvVectorMatrix: ...
    def correct(self) -> None: ...


# ---------------------------------------------------------------------------
# Build: register initialisation steps provided by pimple model
# ---------------------------------------------------------------------------


@pimple.build
def build(self: object) -> list[object]:  # noqa: C901
    """Register all field and model initialisation steps for VoF PIMPLE algorithm."""

    # ------------------------------------------------------------------ #
    # Factory functions (closures) for each field / model                 #
    # ------------------------------------------------------------------ #

    def _gravity_ref(mesh: Any) -> tuple[Any, Any]:
        """Return ``(g, ghRef)`` — gravity field and its reference head."""
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        return g, pyf.dimensionedScalar("ghRef", g.dimensions() * pyf.dimLength, 0.0)

    def create_gh(context: dict[str, Any]) -> volScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gravity_ref(mesh)
        return volScalarField(pyf.Word("gh"), (g & mesh.C()) - gh_ref_dim)

    def create_ghf(context: dict[str, Any]) -> surfaceScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gravity_ref(mesh)
        return surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - gh_ref_dim)

    def create_p(context: dict[str, Any]) -> volScalarField:
        p_rgh = context["fields.p_rgh"]
        rho = context["fields.rho"]
        gh = context["fields.gh"]
        # Absolute pressure: p = p_rgh + rho*gh
        return volScalarField(pyf.Word("p"), p_rgh + rho * gh)

    def create_p_rgh(context: dict[str, Any]) -> volScalarField:
        mesh = context["mesh"]
        mesh.setFluxRequired(pyf.Word("p_rgh"))
        return volScalarField.read_field(mesh, "p_rgh")

    def create_pressure_reference(context: dict[str, Any]) -> dict[str, Any]:
        """Set reference cell/value for p_rgh."""
        p = context["fields.p"]
        p_rgh = context["fields.p_rgh"]
        fv_solution = pyf.dictionary.read("system/fvSolution")
        algo_dict = fv_solution.subDict("PIMPLE")
        # Two-field setRefCell(p, p_rgh, dict) — the stub only knows the
        # single-field overload.
        pRefCell, pRefValue = pyf.setRefCell(p, p_rgh, algo_dict, False)  # type: ignore[call-arg, arg-type]
        return {"pRefCell": pRefCell, "pRefValue": pRefValue}

    # ------------------------------------------------------------------ #
    # Init step list (dependency-ordered)                                 #
    # ------------------------------------------------------------------ #

    return [
        read_vol_field(volVectorField, "U"),
        field("p_rgh", create_p_rgh, depends_on=["mesh"]),
        field("gh", create_gh, depends_on=["mesh"]),
        field("ghf", create_ghf, depends_on=["mesh"]),
        field(
            "p",
            create_p,
            depends_on=["fields.p_rgh", "fields.rho", "fields.gh"],
        ),
        model("pimple_control", create_pimple_control, depends_on=["mesh"]),
        model("cumulativeContErr", lambda _: [0.0]),
        model(
            "pressure_reference",
            create_pressure_reference,
            depends_on=["fields.p", "fields.p_rgh", "mesh"],
        ),
    ]


# ---------------------------------------------------------------------------
# Inner loop predicate
# ---------------------------------------------------------------------------


def inner_loop(ctx: Context) -> bool:
    pimple_control = ctx.models["pimple_control"]
    looping = bool(pimple_control.loop(ctx))
    if looping:
        # Mirror pimpleControl::loop(): on the final outer iteration flag the
        # mesh so fvMatrix::solve picks the <field>Final settings for the
        # library solves buried in turbulence.correct() (k/epsilon/nuTilda).
        ctx.mesh.setFinalIteration(pimple_control.finalIter())
    return looping


# ---------------------------------------------------------------------------
# Helper: alias a registered operation under a different name
# ---------------------------------------------------------------------------


def _alias_operation(
    op_func: Callable[..., FieldUpdates],
    *,
    operation_name: str,
    depends_on: list[str],
) -> Operation:
    return Operation(
        func=SequentialOp(op_func),
        metadata=OperationMetadata(
            op_name=operation_name,
            depends_on=depends_on,
            before=[],
            shape="box",
            color="lightblue",
        ),
    )


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@pimple.operation(operation_number="2.1")
def momentum(
    U: volVectorField,
    rho: volScalarField,
    rhoPhi: surfaceScalarField,
    p_rgh: volScalarField,
    gh: volScalarField,
    ghf: surfaceScalarField,
    pimple_control: Annotated[PimpleControl, "models"],
    mixture: Annotated[MixtureProtocol, "models"],
    turbulence: Annotated[TwoPhaseTransportProtocol, "models"],
) -> FieldUpdates:
    """Density-weighted momentum predictor with surface tension."""
    mesh = U.mesh()

    UEqn = fvVectorMatrix(
        fvm.ddt(rho, U) + fvm.div(rhoPhi, U) + turbulence.divDevRhoReff(rho, U)
    )
    UEqn.relax()

    if pimple_control.momentumPredictor():
        pyf.solve(
            UEqn
            + fvc.reconstruct(
                (
                    mixture.surfaceTensionForce()
                    - ghf * fvc.snGrad(rho)
                    - fvc.snGrad(p_rgh)
                )
                * mesh.magSf()
            )
        )

    return FieldUpdates({"UEqn": UEqn, "U": U})


@pimple.operation(operation_number="2.2", depends_on=["momentum"])
def continuity(
    U: volVectorField,
    rho: volScalarField,
    p_rgh: volScalarField,
    p: volScalarField,
    phi: surfaceScalarField,
    gh: volScalarField,
    ghf: surfaceScalarField,
    UEqn: fvVectorMatrix,
    pimple_control: Annotated[PimpleControl, "models"],
    cumulativeContErr: Annotated[list[float], "models"],
    pressure_reference: Annotated[dict[str, Any], "models"],
    mixture: Annotated[MixtureProtocol, "models"],
) -> FieldUpdates:
    """Pressure-velocity coupling for VoF with surface tension and gravity."""
    pRefCell = pressure_reference["pRefCell"]
    pRefValue = pressure_reference["pRefValue"]
    mesh = U.mesh()

    while pimple_control.correct():
        rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
        rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))

        HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p_rgh))

        phiHbyA = surfaceScalarField(
            pyf.Word("phiHbyA"),
            fvc.flux(HbyA) + fvc.interpolate(rho * rAU) * fvc.ddtCorr(U, phi),
        )

        # Surface tension + gravity contribution on faces
        phig = surfaceScalarField(
            pyf.Word("phig"),
            (mixture.surfaceTensionForce() - ghf * fvc.snGrad(rho))
            * rAUf
            * mesh.magSf(),
        )
        phiHbyA.assign(phiHbyA + phig)

        pyf.adjustPhi(phiHbyA, U, p_rgh)
        pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf)

        while pimple_control.correctNonOrthogonal():
            pEqn = fvScalarMatrix(fvm.laplacian(rAUf, p_rgh) - fvc.div(phiHbyA))
            pEqn.setReference(pRefCell, pRefValue, False)
            pEqn.solve(p_rgh.select(pimple_control.finalInnerIter()))

            if pimple_control.finalNonOrthogonalIter():
                phi.assign(phiHbyA - pEqn.flux())

        # Reconstruct velocity with surface tension + pressure flux correction
        U.assign(HbyA + rAU * fvc.reconstruct((phig - pEqn.flux()) / rAUf))
        U.correctBoundaryConditions()

        # Update absolute pressure from dynamic pressure
        p.assign(p_rgh + rho * gh)

        sum_local, global_err = pyf.computeContinuityErrors(phi)
        cumulativeContErr[0] += global_err
        pyf.Info(
            f"time step continuity errors : sum local = {sum_local}, "
            f"global = {global_err}, cumulative = {cumulativeContErr[0]}"
        )

    return FieldUpdates({"U": U, "p": p, "p_rgh": p_rgh, "phi": phi})


# ---------------------------------------------------------------------------
# Operation collection: expose inner_loop + momentum/continuity aliases
# ---------------------------------------------------------------------------


@pimple.operation_collection
def collected_operations(self: object) -> Operations:
    """Wrap momentum/continuity inside the PIMPLE inner loop.

    The operation-collection path bypasses the spec's default operation
    wrapping, so momentum/continuity are wrapped with dependency resolution
    here (``self`` is the bound runtime) — mirroring the incompressibleFluid
    PIMPLE model.
    """
    model_ops = Operations()

    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            metadata=OperationMetadata(op_name="inner_loop"),
        )
    )

    wrapped_momentum = wrap_with_dependency_resolution(
        momentum, self, pimple._dependency_resolver
    )
    wrapped_continuity = wrap_with_dependency_resolution(
        continuity, self, pimple._dependency_resolver
    )

    model_ops.add(
        _alias_operation(wrapped_momentum, operation_name="momentum", depends_on=[])
    )
    model_ops.add(
        _alias_operation(
            wrapped_continuity, operation_name="continuity", depends_on=["momentum"]
        )
    )

    return model_ops
