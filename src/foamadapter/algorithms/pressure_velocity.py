# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Pressure-velocity coupling algorithms."""

from typing import Any, Literal, Protocol, runtime_checkable

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
from pydantic import BaseModel

from foamadapter.core.plugin_system import PluginSystem
from foamadapter.framework.context import (
    FieldUpdates,
    Model as ModelAnnotation,
)
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model
from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
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


# ============================================================================
# PluginSystem-based Algorithm Registry
# ============================================================================


@PluginSystem.register(discriminator_variable="config", discriminator="algorithm_type")
class PressureVelocityAlgorithmConfig(BaseModel):
    """
    Base class for pressure-velocity coupling algorithm configurations.

    Provides extensibility for different algorithms (SIMPLE, PISO, PIMPLE)
    using the PluginSystem pattern.
    """

    model_config = {"arbitrary_types_allowed": True}

    @property
    def provides(self) -> list[str]:
        """Fields this algorithm provides."""
        return []  # Override in subclasses if needed

    @property
    def requires(self) -> list[str]:
        """Fields this algorithm requires."""
        return []  # Override in subclasses if needed

    @classmethod
    def create(cls, *, config: dict[str, Any]) -> Any:
        """Factory classmethod to create algorithm config from config dict.

        Implemented explicitly for type safety. Calls plugin_model generated
        by @PluginSystem.register decorator.
        """
        return cls.plugin_model(config=config)  # type: ignore[attr-defined]

    def create_algorithm(
        self, pRefCell: int | None = None, pRefValue: float | None = None
    ) -> Any:
        """
        Create the algorithm instance.

        Args:
            pRefCell: Reference cell for pressure
            pRefValue: Reference value for pressure

        Returns:
            Algorithm instance
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement create_algorithm()"
        )

    def setup(
        self, builder: Any, pRefCell: int | None = None, pRefValue: float | None = None
    ) -> Any:
        """
        Setup algorithm and register fields with builder.

        Default implementation just calls create_algorithm().
        """
        return self.create_algorithm(pRefCell, pRefValue)


@PressureVelocityAlgorithmConfig.register
class PimpleConfig(BaseModel):
    """PIMPLE algorithm configuration."""

    algorithm_type: Literal["PIMPLE"] = "PIMPLE"
    model_config = {"arbitrary_types_allowed": True}

    @property
    def provides(self) -> list[str]:
        return []  # Algorithm itself doesn't provide fields during setup

    @property
    def requires(self) -> list[str]:
        return []  # No setup-time dependencies

    def create_algorithm(
        self, pRefCell: int | None = None, pRefValue: float | None = None
    ) -> "PimpleAlgorithm":
        """Create PIMPLE algorithm instance."""
        return PimpleAlgorithm(pRefCell=pRefCell, pRefValue=pRefValue)

    def setup(
        self, builder: Any, pRefCell: int | None = None, pRefValue: float | None = None
    ) -> "PimpleAlgorithm":
        return self.create_algorithm(pRefCell, pRefValue)


@PressureVelocityAlgorithmConfig.register
class SimpleConfig(BaseModel):
    """SIMPLE algorithm configuration (not yet implemented)."""

    algorithm_type: Literal["SIMPLE"] = "SIMPLE"
    model_config = {"arbitrary_types_allowed": True}

    def create(
        self, pRefCell: int | None = None, pRefValue: float | None = None
    ) -> Any:
        """Create SIMPLE algorithm instance."""
        raise NotImplementedError("SIMPLE algorithm not yet implemented")

    def setup(
        self, builder: Any, pRefCell: int | None = None, pRefValue: float | None = None
    ) -> Any:
        raise NotImplementedError("SIMPLE algorithm not yet implemented")


@PressureVelocityAlgorithmConfig.register
class PisoConfig(BaseModel):
    """PISO algorithm configuration (not yet implemented)."""

    algorithm_type: Literal["PISO"] = "PISO"
    model_config = {"arbitrary_types_allowed": True}

    def create(
        self, pRefCell: int | None = None, pRefValue: float | None = None
    ) -> Any:
        """Create PISO algorithm instance."""
        raise NotImplementedError("PISO algorithm not yet implemented")

    def setup(
        self, builder: Any, pRefCell: int | None = None, pRefValue: float | None = None
    ) -> Any:
        raise NotImplementedError("PISO algorithm not yet implemented")


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
        self,
        U: Any,
        phi: Any,
        p: Any,
        turbulence: Any,
        pimple_control: ModelAnnotation[Any],
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
        self, U: Any, p: Any, phi: Any, UEqn: Any, pimple_control: ModelAnnotation[Any]
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
