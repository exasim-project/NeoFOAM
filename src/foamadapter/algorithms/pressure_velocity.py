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
class PressureVelocityAlgorithmProtocol(Protocol):
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
@Model
class PressureVelocityAlgorithm(BaseModel):
    """
    Base class for pressure-velocity coupling algorithms.

    Loads fvSolution, detects algorithm type, and dispatches to registered method.
    """

    model_config = {"arbitrary_types_allowed": True}
    algorithm_type: str | None = None

    @Model.load
    def load_fv_solution(self, path: str = "system/fvSolution") -> None:
        """Read fvSolution and detect algorithm type."""
        fv_solution = pyf.dictionary.read(path)

        # Detect algorithm type from available subdictionaries
        toc = fv_solution.toc()
        if "PIMPLE" in toc:
            self.algorithm_type = "PIMPLE"
        elif "PISO" in toc:
            self.algorithm_type = "PISO"
        elif "SIMPLE" in toc:
            self.algorithm_type = "SIMPLE"
        else:
            raise ValueError("No algorithm found in fvSolution")

    @classmethod
    def from_fv_solution(cls, path: str = "system/fvSolution") -> Any:
        """Factory: create instance, trigger load, return discriminated subclass."""
        # Create instance and trigger LOAD stage
        instance = cls()
        instance.load_fv_solution(path)

        # Use PluginSystem to instantiate correct registered method
        return cls.create(config={"algorithm_type": instance.algorithm_type})

    @property
    def provides(self) -> list[str]:
        """Fields this algorithm provides during setup."""
        return ["p", "U", "phi"]  # Base fields all algorithms provide

    @property
    def requires(self) -> list[str]:
        """Fields this algorithm requires."""
        return []  # Override in subclasses if needed

    @classmethod
    def create(cls, *, config: dict[str, Any]) -> Any:
        """Factory classmethod to create algorithm from config dict.

        Implemented explicitly for type safety. Calls plugin_model generated
        by @PluginSystem.register decorator.
        """
        return cls.plugin_model(config=config)  # type: ignore[attr-defined]

    def setup(self) -> list[Any]:
        """
        Return LazyInit objects for fields this algorithm manages.

        Default implementation registers p, U, phi fields.
        Override to add algorithm-specific fields.

        Returns:
            List of LazyInit objects for field initialization
        """
        from foamadapter.foam.initialization import read_vol_field
        from foamadapter.framework.initialization.helpers import field

        def create_phi(context: dict[str, Any]) -> Any:
            U = context["fields.U"]
            return pyf.createPhi(U)

        return [
            read_vol_field(volScalarField, "p"),
            read_vol_field(volVectorField, "U"),
            field("phi", create_phi, depends_on=["fields.U"]),
        ]


@PressureVelocityAlgorithm.register
@Model
class PimpleMethod(BaseModel):
    """PIMPLE algorithm - unified config and implementation."""

    algorithm_type: Literal["PIMPLE"] = "PIMPLE"
    model_config = {"arbitrary_types_allowed": True}

    # Settings from fvSolution
    nCorrectors: int = 2
    nNonOrthogonalCorrectors: int = 0
    momentumPredictor: bool = True

    # Reference cell/value
    pRefCell: int | None = None
    pRefValue: float | None = None

    # Internal state
    _ops: OperationCollection | None = None

    @Model.load
    def load_fv_solution(self) -> None:
        """Read PIMPLE-specific settings from fvSolution."""
        fv_solution = pyf.dictionary.read("system/fvSolution")
        pimple_dict = fv_solution.subDict("PIMPLE")

        try:
            self.nCorrectors = pimple_dict.get[int]("nCorrectors")
        except (KeyError, AttributeError):
            pass  # Use defaults

        try:
            self.nNonOrthogonalCorrectors = pimple_dict.get[int](
                "nNonOrthogonalCorrectors"
            )
        except (KeyError, AttributeError):
            pass

        try:
            self.momentumPredictor = pimple_dict.get[bool]("momentumPredictor")
        except (KeyError, AttributeError):
            pass

    @property
    def provides(self) -> list[str]:
        return [
            "p",
            "U",
            "phi",
            "pimple_control",
        ]  # PIMPLE provides base fields + control model

    @property
    def requires(self) -> list[str]:
        return []  # No setup-time dependencies

    def setup(self) -> list[Any]:
        """Register p, U, phi fields and pimple_control model."""
        from foamadapter.framework.initialization.helpers import model

        # Get base fields (p, U, phi) from parent class
        initializers = PressureVelocityAlgorithm.setup(self)

        # Add PIMPLE-specific control as a model (not field)
        def create_pimple_control(context: dict[str, Any]) -> Any:
            mesh = context["mesh"]
            return pyf.pimpleControl(mesh)

        initializers.append(
            model("pimple_control", depends_on=["mesh"], create=create_pimple_control)
        )

        return initializers

    def name(self) -> str:
        return "PIMPLE"

    def create_control(self, mesh: Any) -> Any:
        return pyf.pimpleControl(mesh)

    def operations(self) -> OperationCollection:
        """Return algorithm-specific operations."""
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
        turbulence: ModelAnnotation[Any],
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
