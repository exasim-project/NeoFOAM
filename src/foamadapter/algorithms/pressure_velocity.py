# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Pressure-velocity coupling algorithms."""

from typing import Any, Literal

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

    @classmethod
    def create(cls, *, config: dict[str, Any]) -> Any:
        """Factory classmethod to create algorithm config from config dict.

        Implemented explicitly for type safety. Calls plugin_model generated
        by @PluginSystem.register decorator.
        """
        return cls.plugin_model(config=config)  # type: ignore[attr-defined]

    @classmethod
    def from_fvSolution(cls, fvSolution: Any) -> Any:
        """
        Detect algorithm type from fvSolution file and create algorithm instance.

        Reads system/fvSolution to determine which algorithm (PIMPLE/PISO/SIMPLE)
        is being used and returns a ready-to-use algorithm instance.

        pRefCell and pRefValue should be set later via set_pressure_reference().

        Args:
            fvSolution: OpenFOAM dictionary object for system/fvSolution

        Returns:
            Algorithm instance (PimpleAlgorithm, PisoAlgorithm, or SimpleAlgorithm)
            ready to use with setup() and operations() methods.

        Raises:
            ValueError: If no supported algorithm found in fvSolution

        Example:
            fvSolution = pyf.dictionary.read("system/fvSolution")
            algorithm = PressureVelocityAlgorithmConfig.from_fvSolution(fvSolution)
            initializers = algorithm.setup()
            # Later, after pressure field exists:
            algorithm.set_pressure_reference(p, mesh, fvSolution)
        """
        if fvSolution.isDict("PIMPLE"):
            algorithm_type = "PIMPLE"
        elif fvSolution.isDict("PISO"):
            algorithm_type = "PISO"
        elif fvSolution.isDict("SIMPLE"):
            algorithm_type = "SIMPLE"
        else:
            raise ValueError(
                "No supported algorithm (PIMPLE/PISO/SIMPLE) found in system/fvSolution"
            )

        wrapper = cls.create(config={"algorithm_type": algorithm_type})
        return wrapper.config  # type: ignore[attr-defined]


@PressureVelocityAlgorithmConfig.register
@Model
class PimpleAlgorithm(BaseModel):
    """PIMPLE algorithm - unified config and execution."""

    algorithm_type: Literal["PIMPLE"] = "PIMPLE"
    pRefCell: int | None = None
    pRefValue: float | None = None
    model_config = {"arbitrary_types_allowed": True}

    _ops: OperationCollection | None = None

    def setup(self) -> list[Any]:
        """Register p, U, phi fields and pimple_control model."""
        from foamadapter.foam.initialization import read_vol_field
        from foamadapter.framework.initialization.helpers import field, model

        def create_phi(context: dict[str, Any]) -> Any:
            U = context["fields.U"]
            return pyf.createPhi(U)

        def create_pimple_control(context: dict[str, Any]) -> Any:
            mesh = context["mesh"]
            return pyf.pimpleControl(mesh)

        return [
            read_vol_field(volScalarField, "p"),
            read_vol_field(volVectorField, "U"),
            field("phi", create_phi, depends_on=["fields.U"]),
            model("pimple_control", depends_on=["mesh"], create=create_pimple_control),
        ]

    def set_pressure_reference(self, p: Any, mesh: Any, fvSolution: Any) -> None:
        """Set pressure reference cell and value from fvSolution.

        Args:
            p: Pressure field
            mesh: Mesh object
            fvSolution: OpenFOAM dictionary object for system/fvSolution
        """
        import pybFoam as pyf

        # Get algorithm-specific subDict
        algo_dict = fvSolution.subDict("PIMPLE")

        # Extract pRefCell and pRefValue
        pRefCell, pRefValue = pyf.setRefCell(p, algo_dict)
        self.pRefCell = pRefCell
        self.pRefValue = pRefValue

        # Set flux required for pressure
        mesh.setFluxRequired(pyf.Word("p"))

    def name(self) -> str:
        return "PIMPLE"

    def operations(self) -> OperationCollection:
        """Return algorithm-specific operations - just momentum and continuity."""
        if self._ops is None:
            funcs = decorated_member_functions(self)
            self._ops = OperationCollection()
            for func in funcs:
                if not hasattr(func, "_metadata"):
                    continue
                if func._metadata.is_condition:
                    op = Operation.create_IterOp(func)
                    self._ops.add(op)
                else:
                    op = Operation.create_SeqOp(func)
                    self._ops.add(op)

        return self._ops

    @Model.condition
    def inner_loop(
        self,
        pimple_control: ModelAnnotation[pyf.pimpleControl],
    ) -> bool:
        return bool(pimple_control.loop())

    @Model.operation(operation_number=1)
    def momentum(
        self,
        U: Any,
        phi: Any,
        p: Any,
        turbulence: Any,
        pimple_control: ModelAnnotation[pyf.pimpleControl],
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

        return FieldUpdates({"UEqn": UEqn, "U": U})

    @Model.operation(operation_number=2)
    def continuity(
        self,
        U: Any,
        p: Any,
        phi: Any,
        UEqn: Any,
        pimple_control: ModelAnnotation[pyf.pimpleControl],
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


@PressureVelocityAlgorithmConfig.register
class SimpleAlgorithm(BaseModel):
    """SIMPLE algorithm (not yet implemented)."""

    algorithm_type: Literal["SIMPLE"] = "SIMPLE"
    pRefCell: int | None = None
    pRefValue: float | None = None
    model_config = {"arbitrary_types_allowed": True}

    def setup(self) -> list[Any]:
        raise NotImplementedError("SIMPLE algorithm not yet implemented")


@PressureVelocityAlgorithmConfig.register
class PisoAlgorithm(BaseModel):
    """PISO algorithm (not yet implemented)."""

    algorithm_type: Literal["PISO"] = "PISO"
    pRefCell: int | None = None
    pRefValue: float | None = None
    model_config = {"arbitrary_types_allowed": True}

    def setup(self) -> list[Any]:
        raise NotImplementedError("PISO algorithm not yet implemented")
