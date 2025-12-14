# IncompressibleFluid Architecture Cleanup - Phase 2

## Summary

Refactor IncompressibleFluid to always initialize core components, move algorithm factory to PluginSystem, use DAG-based initialization where components register their fields and dependencies, and remove redundant field storage from the solver class.

---

## Changes Overview

| Changed Files | Changed Class | Description | Reason |
|--------------|---------------|-------------|--------|
| `pressure_velocity.py` | `PressureVelocityAlgorithmConfig` | Add PluginSystem-based algorithm registry | Move factory logic out of solver, enable extensible algorithm selection |
| `pressure_velocity.py` | `PimpleConfig`, `SimpleConfig`, `PisoConfig` | Add algorithm configuration classes | Each algorithm type registers with PluginSystem |
| `pressure_velocity.py` | `PimpleAlgorithm` | Add `setup()` method with field dependencies | Component registers its fields and dependencies for DAG ordering |
| `incompressibleFluid.py` | `TransportModel` | Add `setup()` method with field dependencies | Component declares what fields it provides and requires |
| `incompressibleFluid.py` | `TurbulenceModel` | Add `setup()` method with field dependencies | Component declares what fields it provides and requires |
| `incompressibleFluid.py` | `IncompressibleFluid` | Remove `_create_algorithm()` method | Factory logic moved to `PressureVelocityAlgorithmConfig` |
| `incompressibleFluid.py` | `IncompressibleFluid` | Remove `mesh`, `runTime`, `p`, `U`, `phi` attributes | Components manage their own context, fields accessed via Context |
| `incompressibleFluid.py` | `IncompressibleFluid` | Remove `create_context()` method | No backward compatibility needed, use `initialize()` directly |
| `incompressibleFluid.py` | `IncompressibleFluid` | Change `_pressure_velocity`, `_transport`, `_turbulence` to non-optional | Always initialized in CONFIGURE, simplifies all logic |
| `incompressibleFluid.py` | `IncompressibleFluid` | Update `configure_solver()` to use PluginSystem | Use DAG-based component initialization |
| `incompressibleFluid.py` | `IncompressibleFluid` | Update `setup_runtime()` to delegate to components | Components call their own `setup()` which registers fields |
| `initialization.py` | `SolverInitializer` | Add DAG-based setup ordering | Components declare dependencies, initializer resolves order automatically |
| `test_incompressible_fluid.py` | Tests | Update all tests to use new API | Remove tests for deprecated methods, access fields via Context |

---

## Architecture Design

### DAG-Based Component Initialization

Each component declares:
1. **`provides`**: List of field names this component adds to context
2. **`requires`**: List of field names this component needs from context
3. **`setup(builder)`**: Method that creates instances and registers fields

The initializer builds a DAG from these declarations and executes setup in topological order.

```python
class ComponentSetup(Protocol):
    """Protocol for components that participate in DAG-based initialization."""

    @property
    def provides(self) -> list[str]:
        """Field names this component adds to context."""
        ...

    @property
    def requires(self) -> list[str]:
        """Field names this component needs from context."""
        ...

    def setup(self, builder: ContextBuilder) -> None:
        """Create instances and register fields with builder."""
        ...
```

### Example: Component Dependencies

```
mesh, runTime (solver core)
    │
    ▼
p, U (solver reads fields)
    │
    ▼
phi (created from U)
    │
    ▼
laminarTransport (requires U, phi)
    │
    ▼
turbulence (requires U, phi, laminarTransport)
    │
    ▼
pimple_control (requires mesh)
```

---

## Implementation Steps

### Step 1: Create PluginSystem-based Algorithm Registry

**File:** `src/foamadapter/algorithms/pressure_velocity.py`

Add base class and registrations:

```python
from abc import abstractmethod
from typing import Any, Literal
from pydantic import BaseModel
from foamadapter.core.plugin_system import PluginSystem

@PluginSystem.register(discriminator_variable="config", discriminator="algorithm_type")
class PressureVelocityAlgorithmConfig(BaseModel):
    """Base class for pressure-velocity coupling algorithm configurations."""
    model_config = {"arbitrary_types_allowed": True}

    @property
    def provides(self) -> list[str]:
        """Fields this algorithm provides."""
        return ["pimple_control"]  # Override in subclasses

    @property
    def requires(self) -> list[str]:
        """Fields this algorithm requires."""
        return ["mesh"]  # Override in subclasses

    @abstractmethod
    def create(self, pRefCell: int | None = None, pRefValue: float | None = None) -> Any:
        """Create the algorithm instance."""
        ...

    @abstractmethod
    def setup(self, builder: Any, pRefCell: int | None, pRefValue: float | None) -> Any:
        """Setup algorithm and register fields with builder."""
        ...


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

    def create(self, pRefCell: int | None = None, pRefValue: float | None = None) -> "PimpleAlgorithm":
        return PimpleAlgorithm(pRefCell=pRefCell, pRefValue=pRefValue)

    def setup(self, builder: Any, pRefCell: int | None, pRefValue: float | None) -> "PimpleAlgorithm":
        return self.create(pRefCell, pRefValue)


@PressureVelocityAlgorithmConfig.register
class SimpleConfig(BaseModel):
    """SIMPLE algorithm configuration (not yet implemented)."""
    algorithm_type: Literal["SIMPLE"] = "SIMPLE"
    model_config = {"arbitrary_types_allowed": True}

    def create(self, pRefCell: int | None = None, pRefValue: float | None = None) -> Any:
        raise NotImplementedError("SIMPLE algorithm not yet implemented")

    def setup(self, builder: Any, pRefCell: int | None, pRefValue: float | None) -> Any:
        raise NotImplementedError("SIMPLE algorithm not yet implemented")


@PressureVelocityAlgorithmConfig.register
class PisoConfig(BaseModel):
    """PISO algorithm configuration (not yet implemented)."""
    algorithm_type: Literal["PISO"] = "PISO"
    model_config = {"arbitrary_types_allowed": True}

    def create(self, pRefCell: int | None = None, pRefValue: float | None = None) -> Any:
        raise NotImplementedError("PISO algorithm not yet implemented")

    def setup(self, builder: Any, pRefCell: int | None, pRefValue: float | None) -> Any:
        raise NotImplementedError("PISO algorithm not yet implemented")
```

---

### Step 2: Add Setup Methods to Transport/Turbulence Models

**File:** `src/foamadapter/solver/incompressibleFluid.py`

Update `TransportModel` and `TurbulenceModel` with DAG-aware setup:

```python
@PluginSystem.register(discriminator_variable="config", discriminator="transport_type")
class TransportModel(BaseModel):
    """Base class for transport property models."""
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
        """Create the transport model instance."""
        ...

    def setup(self, builder: Any) -> Any:
        """Create instance and register with builder."""
        U = builder.get_field("U")
        phi = builder.get_field("phi")
        instance = self.create(U, phi)
        builder.add_field("laminarTransport", instance)
        return instance


@PluginSystem.register(discriminator_variable="config", discriminator="turbulence_type")
class TurbulenceModel(BaseModel):
    """Base class for turbulence models."""
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
        """Create the turbulence model instance."""
        ...

    def setup(self, builder: Any) -> Any:
        """Create instance and register with builder."""
        U = builder.get_field("U")
        phi = builder.get_field("phi")
        transport = builder.get_field("laminarTransport")
        instance = self.create(U, phi, transport)
        builder.add_field("turbulence", instance)
        return instance
```

---

### Step 3: Simplify IncompressibleFluid

**File:** `src/foamadapter/solver/incompressibleFluid.py`

Remove redundant attributes and methods:

```python
@Solver
class IncompressibleFluid(BaseModel):
    """Incompressible fluid solver with modular physics."""

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

    # === Core Components (always initialized, never None after CONFIGURE) ===
    _pressure_velocity: Any = None  # Set in configure_solver
    _transport: Any = None          # Set in setup_runtime
    _turbulence: Any = None         # Set in setup_runtime

    # === Optional Physics Models ===
    models: list[IncompressibleFluidModel] = []

    # REMOVED: mesh, runTime, p, U, phi - accessed via Context
    # REMOVED: _create_algorithm() - use PressureVelocityAlgorithmConfig
    # REMOVED: create_context() - no backward compatibility needed
```

---

### Step 4: Update configure_solver to Use PluginSystem

```python
@Solver.configure
def configure_solver(self, registry: ModelRegistry) -> None:
    """CONFIGURE: Create algorithm using PluginSystem."""
    from foamadapter.algorithms.pressure_velocity import PressureVelocityAlgorithmConfig

    # Create algorithm using PluginSystem
    algo_wrapper = PressureVelocityAlgorithmConfig.create(
        config={"algorithm_type": self.algorithm}
    )
    self._pressure_velocity = algo_wrapper.config.create(
        pRefCell=self.pRefCell,
        pRefValue=self.pRefValue
    )
    registry.register("algorithm", self._pressure_velocity)

    self.configured = True
```

---

### Step 5: Update setup_runtime to Use DAG-based Component Setup

```python
@Solver.setup
def setup_runtime(self, mesh: Any, builder: Any) -> None:
    """SETUP: Initialize runtime and let components setup via DAG order."""
    # Create runtime and mesh (solver core responsibility)
    argList = pyf.argList(self.argv)
    runTime = pyf.Time(argList)
    mesh = pyf.fvMesh(runTime)

    builder.set_mesh(mesh)
    builder.set_runtime(runTime)

    # Read core fields (solver responsibility)
    p = volScalarField.read_field(mesh, "p")
    U = volVectorField.read_field(mesh, "U")
    phi = pyf.createPhi(U)

    builder.add_field("p", p)
    builder.add_field("U", U)
    builder.add_field("phi", phi)

    # Create component configs
    transport_wrapper = TransportModel.create(config={"transport_type": self.transport_type})
    turbulence_wrapper = TurbulenceModel.create(config={"turbulence_type": self.turbulence_type})

    # Components setup themselves (DAG order handled by initializer)
    self._transport = transport_wrapper.config.setup(builder)
    self._turbulence = turbulence_wrapper.config.setup(builder)

    # Read fvSolution and set reference cell
    fvSolution = pyf.dictionary.read("system/fvSolution")
    pRefCell, pRefValue = pyf.setRefCell(p, fvSolution.subDict("PIMPLE"))
    mesh.setFluxRequired(pyf.Word("p"))

    # Update algorithm with reference cell/value
    self._pressure_velocity.pRefCell = pRefCell
    self._pressure_velocity.pRefValue = pRefValue

    self.setup_complete = True
```

---

### Step 6: Update run() Method

```python
def run(self) -> None:
    """Run the complete simulation using 3-stage initialization."""
    from foamadapter.framework.initialization import SolverInitializer

    # 3-stage initialization returns Context directly
    initializer = SolverInitializer(self)
    ctx = initializer.initialize(mesh=None)

    # Setup models (algorithm control)
    ops = self.operations()
    ops["setup_models"].run(ctx)

    Info("Starting time loop")

    # Run main loop
    self.main_loop(ctx)

    Info("End")
```

---

### Step 7: Simplify operations() and main_loop()

Remove all `if self._pressure_velocity is None` checks since components are always initialized:

```python
def operations(self, domain_name: str | None = None) -> OperationCollection:
    """Collect all operations from solver and models."""
    _ = domain_name

    # Collect solver operations
    funcs = decorated_member_functions(self)
    ops = OperationCollection()
    for func in funcs:
        op = Operation.create_SeqOp(func)
        ops.add(op)

    # Add algorithm operations (always initialized after CONFIGURE)
    algo_ops = self._pressure_velocity.operations()
    ops.add(algo_ops)

    # Add optional model operations
    for model in self.models:
        if hasattr(model, 'operations'):
            model_ops = model.operations()
            ops.add(model_ops)

    return ops


def main_loop(self, ctx: Context) -> None:
    """Main simulation loop - algorithm agnostic!"""
    ops = self.operations()

    # Build the main execution graph
    main_loop = StepBuilder()

    time_loop_op = Operation(
        func=IterativeOp(CFLCondition(self.maxDeltaT)),
        operation_name="time_loop",
        operation_number=1,
    )

    with main_loop.loop(time_loop_op) as time_loop:
        time_loop.step(ops["print_time"])

        algo_ops = self._pressure_velocity.operations()
        time_loop.step(algo_ops["momentum"])
        time_loop.step(algo_ops["continuity"])

        time_loop.step(ops["turbulence_correction"])
        time_loop.step(ops["write_output"])

    main_loop.operations.run(ctx)
```

---

### Step 8: Update Tests

**File:** `test/initialization/test_incompressible_fluid.py`

1. Remove tests for `create_context()` (deleted method)
2. Remove tests accessing `solver.mesh`, `solver.p`, etc. (deleted attributes)
3. Update assertions to check `_pressure_velocity is not None` after CONFIGURE
4. Access fields via `context.fields["p"]` instead of `solver.p`

---

## Future Enhancements

### DAG Visualization

The initialization DAG could be visualized for debugging:

```
solver.initialize() builds DAG:
  mesh, runTime → p, U → phi → laminarTransport → turbulence
```

### Component Discovery

Components could be auto-discovered from models list:

```python
def _collect_setup_components(self) -> list[ComponentSetup]:
    """Collect all components that need setup."""
    components = []

    # Transport and turbulence (from type config)
    components.append(self._get_transport_config())
    components.append(self._get_turbulence_config())

    # Optional models that implement ComponentSetup
    for model in self.models:
        if hasattr(model, 'provides') and hasattr(model, 'requires'):
            components.append(model)

    return components
```

---

## Testing Checklist

- [ ] All 27 existing tests pass (with updates for new API)
- [ ] Algorithm created via PluginSystem in CONFIGURE
- [ ] Transport/Turbulence setup via component `setup()` methods
- [ ] No None checks needed for `_pressure_velocity`, `_transport`, `_turbulence`
- [ ] Fields accessible only via Context (not solver attributes)
- [ ] DAG ordering respects component dependencies
- [ ] NotImplementedError raised for SIMPLE/PISO algorithms
