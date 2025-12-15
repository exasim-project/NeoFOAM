# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from abc import abstractmethod
from typing import Any, Literal

import pybFoam as pyf  # type: ignore[import-not-found]
from pybFoam import (
    Info,
    volScalarField,
    volVectorField,
)
from pybFoam.turbulence import incompressibleTurbulenceModel, singlePhaseTransportModel  # type: ignore[import-not-found]
from pydantic import BaseModel

from foamadapter.algorithms.pressure_velocity import (
    PressureVelocityAlgorithmConfig,
)
from foamadapter.core.plugin_system import PluginSystem
from foamadapter.framework.context import (
    Context,
    FieldUpdates,
    Model as ModelAnnotation,
)
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.initialization import ConfigContext
from foamadapter.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
)
from foamadapter.framework.solver import Solver


class CFLCondition:
    """Condition for CFL-based time stepping."""

    def __init__(self, maxDeltaT: float) -> None:
        self.GREAT = 1e30
        self.SMALL = 1e-15
        controlDict = pyf.dictionary.read("system/controlDict")
        self.adjustable = controlDict.get[bool]("adjustTimeStep")
        self.maxCFL = controlDict.get[float]("maxCo")
        self.maxDeltaT = maxDeltaT
        self.maxRatio = 1.2

    def __call__(self, ctx: Context) -> bool:
        """Adjust time step based on CFL number and continue."""
        runTime = ctx.runTime
        phi = ctx.fields["phi"]

        if not self.adjustable:
            runTime.increment()
            return runTime.loop()

        deltaT = runTime.deltaTValue()
        maxCFLNumber, meanCFLNumber = pyf.computeCFLNumber(phi)
        Info(f"Courant Number mean: {meanCFLNumber}, max: {maxCFLNumber}")

        ratio = self.maxCFL / maxCFLNumber if maxCFLNumber > self.SMALL else self.GREAT
        ratio = min(ratio, self.maxRatio)

        if ratio > self.SMALL and ratio < self.GREAT:
            finalDeltaT = min(deltaT * ratio, self.maxDeltaT)
            runTime.setDeltaT(finalDeltaT)

        runTime.increment()
        return runTime.loop()


# ============================================================================
# Core Component Base Classes (IncompressibleFluid-specific, extensible)
# ============================================================================


@PluginSystem.register(discriminator_variable="config", discriminator="transport_type")
class TransportModel(BaseModel):
    """
    Base class for transport property models.

    Provides extensibility for different transport models (single-phase,
    two-phase, non-Newtonian, etc.) within the IncompressibleFluid solver.
    """

    model_config = {"arbitrary_types_allowed": True}

    @property
    def provides(self) -> list[str]:
        """Fields this transport model adds to context."""
        return ["laminarTransport"]

    @property
    def requires(self) -> list[str]:
        """Fields this transport model needs."""
        return ["U", "phi"]

    @abstractmethod
    def create(self, U: Any, phi: Any) -> Any:
        """
        Create the transport model instance.

        Args:
            U: Velocity field
            phi: Face flux field

        Returns:
            Transport model object
        """
        ...

    def setup(self, builder: Any) -> Any:
        """
        Create instance and register with builder.

        Uses builder.get_field() to retrieve required fields,
        then calls create() and adds result to builder.
        """
        U = builder.get_field("U")
        phi = builder.get_field("phi")
        instance = self.create(U, phi)
        builder.add_field("laminarTransport", instance)
        return instance


@TransportModel.register
class SinglePhaseTransport(BaseModel):
    """Single-phase Newtonian transport properties (default)."""

    transport_type: Literal["singlePhase"] = "singlePhase"
    model_config = {"arbitrary_types_allowed": True}

    def create(self, U: Any, phi: Any) -> Any:
        """Create single-phase transport model using OpenFOAM."""
        return singlePhaseTransportModel(U, phi)


@PluginSystem.register(discriminator_variable="config", discriminator="turbulence_type")
class TurbulenceModel(BaseModel):
    """
    Base class for turbulence models.

    Provides extensibility for different turbulence models (OpenFOAM RTS,
    laminar, custom implementations) within the IncompressibleFluid solver.
    """

    model_config = {"arbitrary_types_allowed": True}

    @property
    def provides(self) -> list[str]:
        """Fields this turbulence model adds to context."""
        return ["turbulence"]

    @property
    def requires(self) -> list[str]:
        """Fields this turbulence model needs."""
        return ["U", "phi", "laminarTransport"]

    @abstractmethod
    def create(self, U: Any, phi: Any, transport: Any) -> Any:
        """
        Create the turbulence model instance.

        Args:
            U: Velocity field
            phi: Face flux field
            transport: Transport model

        Returns:
            Turbulence model object
        """
        ...

    def setup(self, builder: Any) -> Any:
        """
        Create instance and register with builder.

        Uses builder.get_field() to retrieve required fields,
        then calls create() and adds result to builder.
        """
        U = builder.get_field("U")
        phi = builder.get_field("phi")
        transport = builder.get_field("laminarTransport")
        instance = self.create(U, phi, transport)
        builder.add_field("turbulence", instance)
        return instance


@TurbulenceModel.register
class OpenFOAMTurbulence(BaseModel):
    """Wrapper for OpenFOAM's turbulence models (default)."""

    turbulence_type: Literal["openfoam"] = "openfoam"
    model_config = {"arbitrary_types_allowed": True}

    def create(self, U: Any, phi: Any, transport: Any) -> Any:
        """Create turbulence model using OpenFOAM's runtime selection."""
        return incompressibleTurbulenceModel.New(U, phi, transport)


@TurbulenceModel.register
class LaminarModel(BaseModel):
    """Explicit laminar (no turbulence) - useful for testing."""

    turbulence_type: Literal["laminar"] = "laminar"
    model_config = {"arbitrary_types_allowed": True}

    def create(self, U: Any, phi: Any, transport: Any) -> Any:
        """
        Create a mock laminar turbulence model for testing.

        Note: This returns a minimal turbulence object. For production use,
        use OpenFOAM's laminar model via openfoam type.
        """
        # For now, return OpenFOAM's turbulence model which will read
        # the laminar model from constant/momentumTransport
        return incompressibleTurbulenceModel.New(U, phi, transport)


# ============================================================================
# Optional Physics Model Base Class
# ============================================================================


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class IncompressibleFluidModel(BaseModel):
    """
    Base class for optional physics models that extend IncompressibleFluid.

    Models implementing this class:
    - Participate in 3-stage initialization (READ_FILES, CONFIGURE, SETUP)
    - Contribute operations to the execution graph
    - Are registered via PluginSystem for type-safe configuration

    Core components (pressure_velocity, transport, turbulence) are NOT
    IncompressibleFluidModels - they have their own base classes above.

    Example models: Buoyancy, Radiation, Species Transport, etc.
    """

    model_config = {"arbitrary_types_allowed": True}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier."""
        ...

    def operations(self) -> OperationCollection:
        """
        Return operations contributed by this model.

        Default implementation discovers @Model.operation decorated methods.
        """
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops


# ============================================================================
# Solver Class
# ============================================================================


@Solver
class IncompressibleFluid(BaseModel):
    """
    Incompressible fluid solver with modular physics.

    Core components (always present, type-configurable):
    - pressure_velocity: Algorithm for pressure-velocity coupling
    - transport: Transport properties model
    - turbulence: Turbulence model

    Optional models (user-added via add_model()):
    - Buoyancy, radiation, species transport, etc.
    """

    model_config = {"arbitrary_types_allowed": True}

    # === Configuration ===
    name: Literal["IncompressibleFluid"] = "IncompressibleFluid"
    argv: list[str] = []

    # Core component type selection
    algorithm: Literal["SIMPLE", "PISO", "PIMPLE"] = "PIMPLE"
    transport_type: Literal["singlePhase"] = "singlePhase"
    turbulence_type: Literal["openfoam", "laminar"] = "openfoam"

    pRefCell: int | None = None
    pRefValue: float | None = None
    maxDeltaT: float = 1e5

    # Lifecycle state tracking
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False

    # === Core Components (private, populated during initialization) ===
    _pressure_velocity: Any | None = None
    _transport: Any | None = None
    _turbulence: Any | None = None
    _algorithm_config: dict[str, str] | None = (
        None  # Algorithm configuration for SETUP stage
    )

    # === Optional Physics Models ===
    models: list[IncompressibleFluidModel] = []

    def add_model(self, model: IncompressibleFluidModel) -> "IncompressibleFluid":
        """
        Add an optional physics model to extend solver capabilities.

        Models participate in the 3-stage initialization lifecycle
        (READ_FILES, CONFIGURE, SETUP) and contribute operations to
        the execution graph.

        Args:
            model: An IncompressibleFluidModel instance (buoyancy, radiation, etc.)

        Returns:
            self for fluent chaining

        Example:
            solver = (
                IncompressibleFluid(argv=["cavity"])
                .add_model(BuoyancyModel(beta=3e-3))
                .add_model(RadiationModel(type="P1"))
            )
        """
        self.models.append(model)
        return self

    def get_models(self) -> list[Any]:
        """
        Return all models owned by this solver.

        Returns models that have lifecycle methods: algorithm and optional models.
        """
        models = []
        # Add algorithm (has lifecycle methods)
        if self._pressure_velocity is not None:
            models.append(self._pressure_velocity)

        # Add optional physics models
        models.extend(self.models)
        return models

    @Solver.load
    def load_control_dict(self) -> None:
        """LOAD: Load solver control settings from configuration files."""
        # Read controlDict
        controlDict = pyf.dictionary.read("system/controlDict")
        try:
            self.maxDeltaT = controlDict.get[float]("maxDeltaT")
        except KeyError:
            pass  # Use default value

        # Read fvSolution for reference cell/value
        # Note: This requires mesh, so actual reading is deferred to BUILD
        # We just mark files as read here
        self.files_read = True

    @Solver.resolve_dependencies
    def configure_solver(self, config: ConfigContext) -> None:
        """RESOLVE_DEPENDENCIES: Validate solver configuration and connect models."""
        # Validate algorithm choice (configuration-time check)
        if self.algorithm not in ["SIMPLE", "PISO", "PIMPLE"]:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Initialize core components - always non-None after RESOLVE_DEPENDENCIES
        # Components will be fully set up in BUILD stage
        self._transport = TransportModel.create(
            config={"transport_type": self.transport_type}
        )
        self._turbulence = TurbulenceModel.create(
            config={"turbulence_type": self.turbulence_type}
        )

        # Algorithm config will be used in BUILD to create instance with pRefCell/pRefValue
        # Store config for later use
        self._algorithm_config = {"algorithm_type": self.algorithm}

        # Register with config context
        config.register("transport", self._transport)
        config.register("turbulence", self._turbulence)

        self.configured = True

    @Solver.build
    def setup_runtime(self, mesh: Any, builder: Any) -> None:
        """BUILD: Initialize runtime structures and fields."""
        # Create runtime and mesh
        argList = pyf.argList(self.argv)
        runTime = pyf.Time(argList)
        mesh = pyf.fvMesh(runTime)

        # Read primary fields
        p = volScalarField.read_field(mesh, "p")
        U = volVectorField.read_field(mesh, "U")
        phi = pyf.createPhi(U)

        # Add primary fields to builder first
        builder.set_mesh(mesh)
        builder.set_runtime(runTime)
        builder.add_field("p", p)
        builder.add_field("U", U)
        builder.add_field("phi", phi)

        # DAG-based component initialization
        # Components must be initialized in RESOLVE_DEPENDENCIES stage first
        if self._transport is None or self._turbulence is None:
            raise RuntimeError(
                "Components not initialized. Call configure_solver() first."
            )

        # Extract the actual config instance from the wrapper and call its create() method directly
        # then add to builder. This bypasses the setup() method approach for now.

        # 1. Transport model (provides: laminarTransport, requires: U, phi)
        transport_config = self._transport.config
        transport_model = transport_config.create(U, phi)
        builder.add_field("laminarTransport", transport_model)

        # 2. Turbulence model (provides: turbulence, requires: U, phi, laminarTransport)
        turbulence_config = self._turbulence.config
        turbulence_model = turbulence_config.create(U, phi, transport_model)
        builder.add_field("turbulence", turbulence_model)

        # 3. Create algorithm with reference cell configuration
        fvSolution = pyf.dictionary.read("system/fvSolution")
        pRefCell, pRefValue = pyf.setRefCell(p, fvSolution.subDict("PIMPLE"))
        mesh.setFluxRequired(pyf.Word("p"))

        # Create algorithm instance using config system
        algorithm_wrapper = PressureVelocityAlgorithmConfig.create(
            config=self._algorithm_config
        )
        self._pressure_velocity = algorithm_wrapper.config.create(pRefCell, pRefValue)

        # Create pimple control
        pimple = pyf.pimpleControl(mesh)
        builder.add_field("pimple", pimple)

        self.setup_complete = True

    @Solver.operation(operation_number=2)
    def setup_models(self, ctx: Context) -> None:
        """Setup algorithm control object."""
        # Remove old pimple from fields if it exists
        ctx.fields.pop("pimple", None)

        # Create algorithm control using algorithm's factory method
        if self._pressure_velocity is None:
            raise RuntimeError(
                "Algorithm model not initialized. Call initialize() first."
            )
        control = self._pressure_velocity.create_control(ctx.mesh)
        ctx.models["pimple_control"] = control

    @Solver.operation(operation_number=3, depends_on=["setup_models"])
    def print_time(self, ctx: Context) -> None:
        """Print current simulation time."""
        runTime = ctx.runTime
        Info(f"Time = {runTime.timeName()}")

    @Solver.operation(operation_number=4, depends_on=["continuity"])
    def turbulence_correction(
        self, laminarTransport, turbulence, pimple_control: ModelAnnotation
    ) -> FieldUpdates:
        """
        Correct turbulence model after pressure-velocity coupling.
        """
        if pimple_control.turbCorr():
            laminarTransport.correct()
            turbulence.correct()

        return FieldUpdates(
            {"laminarTransport": laminarTransport, "turbulence": turbulence}
        )

    @Solver.operation(operation_number=5, depends_on=["turbulence_correction"])
    def write_output(self, ctx: Context) -> None:
        """Write fields to disk."""
        runTime = ctx.runTime
        runTime.write(True)
        runTime.printExecutionTime()

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        """
        Collect all operations from solver and models.

        Operations are collected from multiple sources and merged into a single
        collection. The execution order is determined by operation numbers and
        dependencies.

        Returns:
            OperationCollection with all operations from:
            - Solver's own @Solver.operation decorated methods
            - Pressure-velocity algorithm
            - Optional physics models (buoyancy, radiation, etc.)
        """
        _ = domain_name  # Part of SolverInterface, unused in this implementation

        # Collect solver operations
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)

        # Add algorithm operations - must be initialized first
        if self._pressure_velocity is None:
            raise RuntimeError(
                "Algorithm not initialized. Call initialize() or run() first."
            )
        algo_ops = self._pressure_velocity.operations()
        ops.add(algo_ops)

        # Add optional model operations
        for model in self.models:
            if hasattr(model, "operations"):
                model_ops = model.operations()
                ops.add(model_ops)

        return ops

    def main_loop(self, ctx: Context) -> None:
        """
        Main simulation loop - algorithm agnostic!

        The algorithm provides momentum and continuity operations that encapsulate
        the specific pressure-velocity coupling strategy.
        """
        if self._pressure_velocity is None:
            raise RuntimeError(
                "Algorithm not initialized. Call initialize() or run() first."
            )

        ops = self.operations()

        # Build the main execution graph
        main_loop = StepBuilder()

        # Time loop
        time_loop_op = Operation(
            func=IterativeOp(CFLCondition(self.maxDeltaT)),
            operation_name="time_loop",
            operation_number=1,
        )

        with main_loop.loop(time_loop_op) as time_loop:
            time_loop.step(ops["print_time"])

            # Algorithm provides momentum and continuity operations
            algo_ops = self._pressure_velocity.operations()
            time_loop.step(algo_ops["momentum"])
            time_loop.step(algo_ops["continuity"])

            # Solver handles turbulence correction
            time_loop.step(ops["turbulence_correction"])

            time_loop.step(ops["write_output"])

        # Execute the operations
        main_loop.operations.run(ctx)

    def run(self) -> None:
        """Run the complete simulation using 3-stage initialization."""
        from foamadapter.framework.initialization import SolverInitializer

        # 3-stage initialization
        initializer = SolverInitializer(self)
        ctx = initializer.initialize(mesh=None)

        # Setup models (algorithm control)
        ops = self.operations()
        ops["setup_models"].run(ctx)

        Info("Starting time loop")

        # Run main loop
        self.main_loop(ctx)

        Info("End")
