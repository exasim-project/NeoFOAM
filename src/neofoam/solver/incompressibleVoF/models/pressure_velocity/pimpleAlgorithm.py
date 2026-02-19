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

from typing import Annotated, Callable, Optional, Protocol

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

from neofoam.algorithms.control import PimpleControl
from neofoam.foam.initialization import read_vol_field
from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field, model
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
)
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
def build() -> list[object]:  # noqa: C901
    """Register all field and model initialisation steps for VoF PIMPLE algorithm."""

    # ------------------------------------------------------------------ #
    # Factory functions (closures) for each field / model                 #
    # ------------------------------------------------------------------ #

    def _gravity_ref(mesh) -> "pyf.dimensionedScalar":
        """Compute the reference gravity-head as a dimensionedScalar."""
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        return g, pyf.dimensionedScalar("ghRef", g.dimensions() * pyf.dimLength, 0.0)

    def create_gh(context: dict) -> volScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gravity_ref(mesh)
        return volScalarField(pyf.Word("gh"), (g & mesh.C()) - gh_ref_dim)

    def create_ghf(context: dict) -> surfaceScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gravity_ref(mesh)
        return surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - gh_ref_dim)

    def create_p(context: dict) -> volScalarField:
        p_rgh = context["fields.p_rgh"]
        rho = context["fields.rho"]
        gh = context["fields.gh"]
        # Absolute pressure: p = p_rgh + rho*gh
        return volScalarField(pyf.Word("p"), p_rgh + rho * gh)

    def create_p_rgh(context: dict) -> volScalarField:
        mesh = context["mesh"]
        mesh.setFluxRequired(pyf.Word("p_rgh"))
        return volScalarField.read_field(mesh, "p_rgh")

    def create_pressure_reference(context: dict) -> dict:
        """Set reference cell/value for p_rgh."""
        p = context["fields.p"]
        p_rgh = context["fields.p_rgh"]
        fv_solution = pyf.dictionary.read("system/fvSolution")
        algo_dict = fv_solution.subDict("PIMPLE")
        pRefCell, pRefValue = pyf.setRefCell(p, p_rgh, algo_dict, False)
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
    return bool(ctx.models["pimple_control"].loop(ctx))


# ---------------------------------------------------------------------------
# Helper: alias a registered operation under a different name
# ---------------------------------------------------------------------------


def _alias_operation(
    op_func: Callable[..., FieldUpdates],
    *,
    operation_name: str,
    depends_on: list[str],
) -> Operation:
    metadata = getattr(op_func, "_metadata", None)
    return Operation(
        func=SequentialOp(op_func),
        operation_number=getattr(metadata, "operation_number", None),
        operation_name=operation_name,
        domain_name=None,
        depends_on=depends_on,
        before=[],
        shape="box",
        color="lightblue",
        level=0,
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
    pressure_reference: Annotated[dict[str, object], "models"],
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
def collected_operations(self, model_state: object) -> Operations:
    model_ops = Operations()

    model_ops.add(
        Operation(
            func=IterativeOp(inner_loop),
            operation_name="inner_loop",
            operation_number=None,
        )
    )
    model_ops.add(_alias_operation(momentum, operation_name="momentum", depends_on=[]))
    model_ops.add(
        _alias_operation(
            continuity, operation_name="continuity", depends_on=["momentum"]
        )
    )

    return model_ops
