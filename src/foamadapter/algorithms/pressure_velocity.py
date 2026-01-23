# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Pressure-velocity coupling algorithms."""

from typing import Any, Callable, Literal

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
from foamadapter.framework.model import Model
from foamadapter.framework.operations import (
    Operation,
    OperationCollection,
)


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

    @classmethod
    def create(cls, *, config: dict[str, Any]) -> Any:
        """Factory classmethod to create algorithm from config dict.

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


@PressureVelocityAlgorithm.register
@Model
class PimpleAlgorithm(BaseModel):
    """PIMPLE algorithm - unified config and execution."""

    algorithm_type: Literal["PIMPLE"] = "PIMPLE"
    pRefCell: int | None = None
    pRefValue: float | None = None
    model_config = {"arbitrary_types_allowed": True}

    _ops: OperationCollection | None = None
    _use_boussinesq: bool = False  # Set by Boussinesq model during resolve_dependencies

    @Model.resolve_dependencies
    def configure_algorithm(self, config: Any) -> None:
        """RESOLVE_DEPENDENCIES: Allow Boussinesq model to configure this algorithm."""
        pass  # Configuration happens via direct flag access from BoussinesqModel

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

        def create_cumulative_cont_err(context: dict[str, Any]) -> Any:
            # Use list to allow mutation (pass by reference for C++ binding)
            return [0.0]

        return [
            read_vol_field(volScalarField, "p"),
            read_vol_field(volVectorField, "U"),
            field("phi", create_phi, depends_on=["fields.U"]),
            model("pimple_control", depends_on=["mesh"], create=create_pimple_control),
            model("cumulativeContErr", create=create_cumulative_cont_err),
        ]

    def set_pressure_reference(
        self, p: Any, mesh: Any, fvSolution: Any, p_rgh: Any = None
    ) -> None:
        """Set pressure reference cell and value from fvSolution.

        Args:
            p: Pressure field
            mesh: Mesh object
            fvSolution: OpenFOAM dictionary object for system/fvSolution
            p_rgh: Optional p_rgh field for Boussinesq mode
        """

        # Get algorithm-specific subDict
        algo_dict = fvSolution.subDict(self.algorithm_type)

        # Extract pRefCell and pRefValue
        # OpenFOAM's setRefCell(field, dict) looks for {field.name()}RefCell/Point.
        # In Boussinesq cases, the dictionary often uses generic "pRefCell/Point"
        # while the field is named "p_rgh".
        pressure_field = p_rgh if p_rgh is not None else p
        field_name = "p_rgh" if p_rgh is not None else "p"

        # Check if specific keys exist for the provided field
        if not (
            algo_dict.found(f"{field_name}RefCell")
            or algo_dict.found(f"{field_name}RefPoint")
        ):
            # Fallback to generic "p" keys if they exist and we are using p_rgh
            if p_rgh is not None and (
                algo_dict.found("pRefCell") or algo_dict.found("pRefPoint")
            ):
                # Use "p" field for lookup but return results for our use
                pRefCell, pRefValue = pyf.setRefCell(p, algo_dict, True)
            else:
                pRefCell, pRefValue = pyf.setRefCell(pressure_field, algo_dict)
        else:
            pRefCell, pRefValue = pyf.setRefCell(pressure_field, algo_dict)

        self.pRefCell = pRefCell
        self.pRefValue = pRefValue

        # Set flux required for pressure fields
        mesh.setFluxRequired(pyf.Word("p"))
        if p_rgh is not None:
            mesh.setFluxRequired(pyf.Word("p_rgh"))

    def name(self) -> str:
        return "PIMPLE"

    def operations(self) -> OperationCollection:
        """Return algorithm-specific operations."""
        if self._ops is None:
            self._ops = OperationCollection()

            # Add inner_loop condition
            # Type: Callable[..., bool]
            inner_loop_callable: Callable[..., bool] = self.inner_loop  # type: ignore[assignment]
            inner_loop_op = Operation.create_IterOp(inner_loop_callable)
            self._ops.add(inner_loop_op)

            # Add momentum and continuity operations based on Boussinesq mode
            if self._use_boussinesq:
                momentum_op = Operation.create_SeqOp(
                    self.momentum_boussinesq, operation_name="momentum"
                )
                continuity_op = Operation.create_SeqOp(
                    self.continuity_boussinesq, operation_name="continuity"
                )
            else:
                momentum_op = Operation.create_SeqOp(self.momentum)
                continuity_op = Operation.create_SeqOp(self.continuity)

            self._ops.add(momentum_op)
            self._ops.add(continuity_op)

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
        turbulence: ModelAnnotation[Any],
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
        cumulativeContErr: ModelAnnotation[list[float]],
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

            # Calculate and print continuity errors
            sumLocal, globalErr = pyf.computeContinuityErrors(phi)
            cumulativeContErr[0] += globalErr
            pyf.Info(
                f"time step continuity errors : sum local = {sumLocal}, "
                f"global = {globalErr}, cumulative = {cumulativeContErr[0]}"
            )

        return FieldUpdates({"U": U, "p": p, "phi": phi})

    @Model.operation(operation_number=1)
    def momentum_boussinesq(
        self,
        U: Any,
        phi: Any,
        p_rgh: Any,
        rhok: Any,
        ghf: Any,
        turbulence: ModelAnnotation[Any],
        pimple_control: ModelAnnotation[pyf.pimpleControl],
    ) -> FieldUpdates:
        """
        PIMPLE momentum with Boussinesq buoyancy.

        Uses p_rgh formulation and buoyancy gradient term.
        """
        # Assemble momentum equation
        mesh = U.mesh()
        UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
        UEqn.relax()

        # Solve with p_rgh and buoyancy gradient
        if pimple_control.momentumPredictor():
            pyf.solve(
                UEqn
                + fvc.reconstruct(
                    (-ghf * fvc.snGrad(rhok) - fvc.snGrad(p_rgh)) * mesh.magSf()
                )
            )

        return FieldUpdates({"UEqn": UEqn, "U": U})

    @Model.operation(operation_number=2)
    def continuity_boussinesq(
        self,
        U: Any,
        p: Any,
        p_rgh: Any,
        phi: Any,
        UEqn: Any,
        rhok: Any,
        gh: Any,
        ghf: Any,
        pimple_control: ModelAnnotation[pyf.pimpleControl],
        cumulativeContErr: ModelAnnotation[list[float]],
    ) -> FieldUpdates:
        """
        PIMPLE continuity with Boussinesq p_rgh formulation.

        Uses buoyancy flux and solves for p_rgh instead of p.
        """
        # Get mesh from U field
        mesh = U.mesh()

        # PIMPLE loop
        while pimple_control.correct():
            # Compute H/A
            rAU = volScalarField(pyf.Word("rAU"), 1.0 / UEqn.A())
            rAUf = surfaceScalarField(pyf.Word("rAUf"), fvc.interpolate(rAU))
            HbyA = volVectorField(pyf.constrainHbyA(rAU * UEqn.H(), U, p_rgh))

            # Buoyancy flux
            phig = surfaceScalarField(
                pyf.Word("phig"), -rAUf * ghf * fvc.snGrad(rhok) * mesh.magSf()
            )

            # Compute flux from H/A with buoyancy
            phiHbyA = surfaceScalarField(
                pyf.Word("phiHbyA"),
                fvc.flux(HbyA) + rAUf * fvc.ddtCorr(U, phi) + phig,
            )

            pyf.constrainPressure(p_rgh, U, phiHbyA, rAUf)

            # Non-orthogonal loop - solve for p_rgh
            while pimple_control.correctNonOrthogonal():
                # Solve pressure equation
                pEqn = fvScalarMatrix(fvm.laplacian(rAUf, p_rgh) - fvc.div(phiHbyA))
                pEqn.setReference(self.pRefCell, self.pRefValue, False)
                pEqn.solve(p_rgh.select(pimple_control.finalInnerIter()))

                # Update flux
                if pimple_control.finalNonOrthogonalIter():
                    phi.assign(phiHbyA - pEqn.flux())

            # Correct velocity with buoyancy
            U.assign(HbyA + rAU * fvc.reconstruct((phig - pEqn.flux()) / rAUf))
            U.correctBoundaryConditions()

            # Update full pressure: p = p_rgh + rhok*gh
            p.assign(p_rgh + rhok * gh)

            # Calculate and print continuity errors
            sumLocal, globalErr = pyf.computeContinuityErrors(phi)
            cumulativeContErr[0] += globalErr
            pyf.Info(
                f"time step continuity errors : sum local = {sumLocal}, "
                f"global = {globalErr}, cumulative = {cumulativeContErr[0]}"
            )

        return FieldUpdates({"U": U, "p": p, "phi": phi})


@PressureVelocityAlgorithm.register
class SimpleAlgorithm(BaseModel):
    """SIMPLE algorithm (not yet implemented)."""

    algorithm_type: Literal["SIMPLE"] = "SIMPLE"
    pRefCell: int | None = None
    pRefValue: float | None = None
    model_config = {"arbitrary_types_allowed": True}

    def setup(self) -> list[Any]:
        raise NotImplementedError("SIMPLE algorithm not yet implemented")


@PressureVelocityAlgorithm.register
class PisoAlgorithm(BaseModel):
    """PISO algorithm (not yet implemented)."""

    algorithm_type: Literal["PISO"] = "PISO"
    pRefCell: int | None = None
    pRefValue: float | None = None
    model_config = {"arbitrary_types_allowed": True}

    def setup(self) -> list[Any]:
        raise NotImplementedError("PISO algorithm not yet implemented")
