# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Pressure-velocity coupling algorithms."""

from typing import Any, Protocol, runtime_checkable

import pybFoam as pyf  # type: ignore[import-not-found]
from pybFoam import (
    fvc,
    fvm,
    fvScalarMatrix,
    fvVectorMatrix,
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from foamadapter.framework.context import (
    Context,
    FieldUpdates,
    Model as ModelAnnotation,
)
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    SequentialOp,
    StepBuilder,
)


@runtime_checkable
class PressureVelocityAlgorithm(Protocol):
    """
    Protocol defining the interface for pressure-velocity coupling algorithms.

    Algorithms expose their operations through the operations() method.
    Operations are accessible by name (e.g., "momentum", "continuity", "turbulence_correction").
    """

    def name(self) -> str:
        """Return algorithm name (SIMPLE, PISO, PIMPLE)."""
        ...

    def create_control(self, mesh: Any) -> Any:
        """
        Create algorithm control object (pimpleControl, pisoControl, simpleControl).

        Args:
            mesh: The finite volume mesh

        Returns:
            Control object for managing algorithm iterations
        """
        ...

    def operations(self) -> OperationCollection:
        """
        Return the algorithm-specific operations.

        Returns:
            Collection of operations used by this algorithm.
            Operations are typically named: "momentum", "continuity"
        """
        ...


@Model
class PimpleAlgorithm:
    """PIMPLE algorithm - self-contained with its own operations."""

    def __init__(self, pRefCell: int | None = None, pRefValue: float | None = None):
        """
        Initialize PIMPLE algorithm.

        Args:
            pRefCell: Reference cell for pressure
            pRefValue: Reference value for pressure
        """
        self.pRefCell = pRefCell
        self.pRefValue = pRefValue
        self._ops: OperationCollection | None = None

    def name(self) -> str:
        return "PIMPLE"

    def create_control(self, mesh: Any) -> Any:
        return pyf.pimpleControl(mesh)

    def operations(self) -> OperationCollection:
        """Return algorithm-specific operations - just momentum and continuity."""
        if self._ops is None:
            funcs = decorated_member_functions(self)
            self._ops = OperationCollection()
            for func in funcs:
                op = Operation.create_SeqOp(func)
                self._ops.add(op)

        return self._ops

    @Model.operation(operation_number=1)
    def momentum(
        self, U, phi, p, turbulence, pimple_control: ModelAnnotation
    ) -> FieldUpdates:
        """
        PIMPLE momentum: Assemble and solve momentum equation.

        Returns a single operation that handles the momentum prediction phase.
        """
        # Assemble momentum equation
        UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
        UEqn.relax()

        # Solve momentum
        if pimple_control.momentumPredictor():
            pyf.solve(UEqn + fvc.grad(p))

        return FieldUpdates({"UEqn": UEqn})

    @Model.operation(operation_number=2)
    def continuity(
        self, U, p, phi, UEqn, pimple_control: ModelAnnotation
    ) -> FieldUpdates:
        """
        PIMPLE continuity: Pressure-velocity coupling with nested loops.

        Returns a single operation that defines the pressure-velocity coupling strategy.

        Structure:
        - PIMPLE outer loop
          - Corrector loop
            - Compute HbyA, flux, adjust
            - Non-orthogonal loop (solve pressure, update flux)
            - Correct velocity
        """
        # PIMPLE loop
        while pimple_control.loop():
            # Corrector loop
            while pimple_control.correct():
                # Compute H/A
                rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
                HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p))

                # Compute flux from H/A
                phiHbyA = surfaceScalarField(
                    pyf.Word("phiHbyA"),
                    fvc.flux(HbyA) + fvc.interpolate(rAU) * fvc.ddtCorr(U, phi),
                )

                # Adjust flux for continuity
                pyf.adjustPhi(phiHbyA, U, p)
                pyf.constrainPressure(p, U, phiHbyA, rAU)

                # Non-orthogonal loop
                while pimple_control.correctNonOrthogonal():
                    # Solve pressure equation
                    pEqn = fvScalarMatrix(fvm.laplacian(rAU, p) - fvc.div(phiHbyA))
                    pEqn.setReference(self.pRefCell, self.pRefValue, False)
                    pEqn.solve(p.select(pimple_control.finalInnerIter()))

                    # Update flux
                    if pimple_control.finalNonOrthogonalIter():
                        phi.assign(phiHbyA - pEqn.flux())

                # Correct velocity
                U.assign(HbyA - rAU * fvc.grad(p))
                U.correctBoundaryConditions()

        return FieldUpdates({"U": U, "p": p, "phi": phi})
