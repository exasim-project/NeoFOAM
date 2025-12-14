# Solver Workflow Improvement Plan

This document describes the current solver workflow, its limitations, and a detailed plan for making solvers more modular and extensible.

---

## Table of Contents

1. [Current Workflow Analysis](#1-current-workflow-analysis)
2. [Problems with Current Design](#2-problems-with-current-design)
3. [Proposed Architecture: Context Creation in SETUP](#3-proposed-architecture-context-creation-in-setup)
4. [Modular Solver Architecture](#4-modular-solver-architecture)
5. [Model Extension System](#5-model-extension-system)
6. [Implementation Plan](#6-implementation-plan)

---

## 1. Current Workflow Analysis

### 1.1 Current Initialization Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRENT WORKFLOW                                  │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Create Solver
    solver = IncompressibleFluid(argv=["cavity"], algorithm="PIMPLE")

Step 2: Initialize (3-stage)
    initializer = SolverInitializer(solver)
    initializer.initialize(mesh=None)

    ├── READ_FILES stage
    │   └── solver.load_control_dict()  → reads controlDict, sets maxDeltaT
    │
    ├── CONFIGURE stage
    │   └── solver.configure_solver(registry)  → creates algorithm_model
    │
    └── SETUP stage
        └── solver.setup_runtime(mesh)  → creates mesh, runTime, p, U, phi
                                        → stores on self.mesh, self.p, etc.

Step 3: Create Context (SEPARATE STEP!)
    ctx = solver.create_context()

    → Creates Context from self.p, self.U, self.phi, self.mesh, self.runTime
    → Now state exists in TWO places: self AND ctx

Step 4: Setup Models
    ops = solver.operations()
    ops["setup_models"].run(ctx)

    → Creates pimple control in ctx.models

Step 5: Run Main Loop
    solver.main_loop(ctx)

    → Executes time loop with operations
```

### 1.2 Current Code Structure

```python
@Solver
class IncompressibleFluid(BaseModel):
    # Configuration (should be immutable)
    algorithm: Literal["SIMPLE", "PISO", "PIMPLE"] = "PIMPLE"
    maxDeltaT: float = 1e5

    # Lifecycle state (mutable)
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False

    # Runtime objects (DUPLICATED in Context later!)
    mesh: Any | None = None
    runTime: Any | None = None
    p: Any | None = None      # Also in ctx.fields["p"]
    U: Any | None = None      # Also in ctx.fields["U"]
    phi: Any | None = None    # Also in ctx.fields["phi"]

    # Models (mixed ownership)
    transport_model: Any | None = None
    turbulence_model: Any | None = None
    algorithm_model: Any | None = None
```

### 1.3 Current Operations Structure

```python
# Solver operations:
@Solver.operation(operation_number=2)
def setup_models(self, ctx: Context) -> None: ...

@Solver.operation(operation_number=3, depends_on=["setup_models"])
def print_time(self, ctx: Context) -> None: ...

@Solver.operation(operation_number=4, depends_on=["continuity"])
def turbulence_correction(self, ...) -> FieldUpdates: ...

@Solver.operation(operation_number=5, depends_on=["turbulence_correction"])
def write_output(self, ctx: Context) -> None: ...

# Algorithm operations (from PimpleAlgorithm):
@Model.operation(operation_number=1)
def momentum(self, ...) -> FieldUpdates: ...

@Model.operation(operation_number=2, depends_on=["momentum"])
def continuity(self, ...) -> FieldUpdates: ...
```

---

## 2. Problems with Current Design

### 2.1 State Duplication

```
Problem: Fields exist in TWO places after initialization

solver.p ──────────────┐
                       ├──► Which is the source of truth?
ctx.fields["p"] ───────┘

If an operation modifies ctx.fields["p"], solver.p is stale.
If solver.p is modified, ctx.fields["p"] is stale.
```

### 2.2 Manual Context Creation

```python
# Current: Easy to forget this step
initializer.initialize(mesh)
ctx = solver.create_context()  # ← MANUAL STEP!
solver.main_loop(ctx)

# Problem: What if user forgets?
initializer.initialize(mesh)
solver.main_loop(ctx)  # ← ctx is undefined!
```

### 2.3 Tight Coupling

```
IncompressibleFluid
    │
    ├── Hardcoded: singlePhaseTransportModel
    ├── Hardcoded: incompressibleTurbulenceModel
    ├── Hardcoded: CFLCondition
    └── Hardcoded: PimpleAlgorithm

Cannot easily:
  - Add buoyancy model
  - Add species transport
  - Add radiation model
  - Change transport model type
  - Use custom time stepping
```

### 2.4 No Model Discovery

```python
# Current: Algorithm is created directly
def _create_algorithm(self) -> PimpleAlgorithm:
    if self.algorithm == "PIMPLE":
        return PimpleAlgorithm(...)
    else:
        raise ValueError("Not implemented")

# Problem: Cannot add new algorithms without modifying solver
```

### 2.5 Monolithic Operations

```python
# Current: All operations defined in one class
class IncompressibleFluid:
    def setup_models(self, ctx): ...
    def print_time(self, ctx): ...
    def turbulence_correction(self, ...): ...
    def write_output(self, ctx): ...

# Problem: Cannot add operations from external models
```

---

## 3. Proposed Architecture: Context Creation in SETUP

### 3.1 New Initialization Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROPOSED WORKFLOW                                 │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Create Solver (configuration only)
    solver = IncompressibleFluid(
        argv=["cavity"],
        algorithm="PIMPLE",
        models=[
            TurbulenceModel(type="kEpsilon"),
            BuoyancyModel(enabled=True),
        ]
    )

Step 2: Initialize (returns Context!)
    ctx = solver.initialize(mesh)

    ├── READ_FILES stage
    │   ├── solver.load_control_dict()
    │   └── [each model].read_files()
    │
    ├── CONFIGURE stage
    │   ├── solver.configure_solver(registry)
    │   └── [each model].configure(registry)
    │
    └── SETUP stage → RETURNS CONTEXT
        ├── [each model].setup(mesh) → contributes to ContextBuilder
        └── solver.setup_runtime(mesh) → finalizes and returns Context

Step 3: Run (Context is ready!)
    solver.run(ctx)
```

### 3.2 ContextBuilder Pattern

```python
class ContextBuilder:
    """Collects contributions from models during SETUP stage."""

    def __init__(self):
        self._fields: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._mesh: Any = None
        self._runTime: Any = None

    def add_field(self, name: str, field: Any) -> None:
        """Add a field to the context."""
        if name in self._fields:
            raise ValueError(f"Field '{name}' already exists")
        self._fields[name] = field

    def add_model(self, name: str, model: Any) -> None:
        """Add a model to the context."""
        self._models[name] = model

    def set_mesh(self, mesh: Any) -> None:
        """Set the mesh (can only be set once)."""
        if self._mesh is not None:
            raise ValueError("Mesh already set")
        self._mesh = mesh

    def set_runtime(self, runTime: Any) -> None:
        """Set the runtime (can only be set once)."""
        if self._runTime is not None:
            raise ValueError("Runtime already set")
        self._runTime = runTime

    def build(self) -> Context:
        """Build the final Context."""
        if self._mesh is None:
            raise ValueError("Mesh not set")
        if self._runTime is None:
            raise ValueError("Runtime not set")

        return Context(
            fields=self._fields,
            models=self._models,
            mesh=self._mesh,
            runTime=self._runTime,
        )
```

### 3.3 Updated SolverInitializer

```python
class SolverInitializer:
    def initialize(self, mesh: Any = None) -> Context:
        """Run 3-stage initialization and return Context."""
        self._run_read_files()
        self._run_configure()
        return self._run_setup(mesh)

    def _run_setup(self, mesh: Any) -> Context:
        """Execute SETUP stage and build Context."""
        builder = ContextBuilder()

        # Models contribute to context
        for model in self._get_models():
            self._execute_setup_with_builder(model, mesh, builder)

        # Solver finalizes context
        self._execute_setup_with_builder(self.solver, mesh, builder)

        return builder.build()

    def _execute_setup_with_builder(
        self, obj: Any, mesh: Any, builder: ContextBuilder
    ) -> None:
        """Execute SETUP methods, passing builder for contributions."""
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            attr = getattr(obj, attr_name, None)
            if callable(attr) and hasattr(attr, "_init_stage"):
                if attr._init_stage == "SETUP":
                    # Pass both mesh and builder
                    attr(mesh, builder)
```

### 3.4 Updated Solver SETUP Method

```python
@Solver
class IncompressibleFluid(BaseModel):
    # Configuration only - NO runtime state!
    algorithm: Literal["SIMPLE", "PISO", "PIMPLE"] = "PIMPLE"
    maxDeltaT: float = 1e5
    pRefCell: int | None = None
    pRefValue: float | None = None

    # Model references (configured during CONFIGURE)
    algorithm_model: PimpleAlgorithm | None = None

    @Solver.setup
    def setup_runtime(self, mesh: Any, builder: ContextBuilder) -> None:
        """SETUP: Create runtime objects and add to context builder."""
        # Create runtime and mesh
        argList = pyf.argList(self.argv)
        runTime = pyf.Time(argList)
        fvMesh = pyf.fvMesh(runTime)

        # Set in builder (not on self!)
        builder.set_mesh(fvMesh)
        builder.set_runtime(runTime)

        # Create fields
        p = volScalarField.read_field(fvMesh, "p")
        U = volVectorField.read_field(fvMesh, "U")
        phi = pyf.createPhi(U)

        # Add to builder (not on self!)
        builder.add_field("p", p)
        builder.add_field("U", U)
        builder.add_field("phi", phi)

        # Create transport/turbulence
        transport = singlePhaseTransportModel(U, phi)
        turbulence = incompressibleTurbulenceModel.New(U, phi, transport)

        builder.add_model("transport", transport)
        builder.add_model("turbulence", turbulence)
        builder.add_model("algorithm", self.algorithm_model)

        # Read fvSolution
        fvSolution = pyf.dictionary.read("system/fvSolution")
        self.pRefCell, self.pRefValue = pyf.setRefCell(
            p, fvSolution.subDict("PIMPLE")
        )
        fvMesh.setFluxRequired(pyf.Word("p"))

        # Update algorithm with reference cell/value
        if self.algorithm_model is not None:
            self.algorithm_model.pRefCell = self.pRefCell
            self.algorithm_model.pRefValue = self.pRefValue
```

### 3.5 New Usage Pattern

```python
# Clean, simple API
solver = IncompressibleFluid(argv=["cavity"])
ctx = solver.initialize()  # Returns ready-to-use Context
solver.run(ctx)

# Or one-liner
IncompressibleFluid(argv=["cavity"]).run()
```

---

## 4. Modular Solver Architecture

### 4.1 Model Composition

```python
@Solver
class IncompressibleFluid(BaseModel):
    """Modular incompressible fluid solver."""

    # Core configuration
    argv: list[str] = []
    algorithm: str = "PIMPLE"

    # Composable models (optional extensions)
    models: list[SolverModel] = []

    def add_model(self, model: SolverModel) -> "IncompressibleFluid":
        """Fluent API to add models."""
        self.models.append(model)
        return self

# Usage:
solver = (
    IncompressibleFluid(argv=["cavity"])
    .add_model(TurbulenceModel(type="kEpsilon"))
    .add_model(BuoyancyModel(beta=3e-3))
    .add_model(RadiationModel(type="P1"))
)
ctx = solver.initialize()
solver.run(ctx)
```

### 4.2 SolverModel Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SolverModel(Protocol):
    """Protocol for models that extend solver capabilities."""

    @property
    def name(self) -> str:
        """Unique name for this model."""
        ...

    def read_files(self) -> None:
        """READ_FILES: Load configuration from files."""
        ...

    def configure(self, registry: ModelRegistry) -> None:
        """CONFIGURE: Validate and connect to other models."""
        ...

    def setup(self, mesh: Any, builder: ContextBuilder) -> None:
        """SETUP: Create runtime objects and add to context."""
        ...

    def operations(self) -> OperationCollection:
        """Return operations this model contributes."""
        ...
```

### 4.3 Example: Buoyancy Model

```python
@Model
class BuoyancyModel(BaseModel):
    """Adds buoyancy effects to incompressible solver."""

    name: str = "buoyancy"
    enabled: bool = True
    beta: float = 3e-3  # Thermal expansion coefficient
    TRef: float = 300.0  # Reference temperature
    g: tuple[float, float, float] = (0, -9.81, 0)

    # Runtime state (populated during SETUP)
    _T: Any = None
    _rhok: Any = None

    @Model.read_files
    def load_buoyancy_properties(self) -> None:
        """Load buoyancy properties from file."""
        try:
            props = pyf.dictionary.read("constant/buoyancyProperties")
            self.beta = props.get[float]("beta")
            self.TRef = props.get[float]("TRef")
        except FileNotFoundError:
            pass  # Use defaults

    @Model.configure
    def configure_buoyancy(self, registry: ModelRegistry) -> None:
        """Connect to transport model."""
        transport = registry.get("transport")
        if transport is None:
            raise ValueError("BuoyancyModel requires transport model")

    @Model.setup
    def setup_buoyancy(self, mesh: Any, builder: ContextBuilder) -> None:
        """Create temperature field and buoyancy term."""
        # Read temperature field
        T = volScalarField.read_field(mesh, "T")
        builder.add_field("T", T)

        # Create rhok = 1 - beta*(T - TRef)
        rhok = volScalarField(
            "rhok",
            mesh,
            1.0 - self.beta * (T - self.TRef)
        )
        builder.add_field("rhok", rhok)
        builder.add_model("buoyancy", self)

    @Model.operation(operation_number=1, depends_on=["momentum"])
    def buoyancy_correction(
        self,
        U: volVectorField,
        rhok: volScalarField,
        ctx: Context
    ) -> FieldUpdates:
        """Add buoyancy source term to momentum equation."""
        g = pyf.dimensionedVector("g", pyf.dimAcceleration, self.g)

        # Modify momentum with buoyancy
        # UEqn += rhok * g

        return FieldUpdates({"U": U})

    def operations(self) -> OperationCollection:
        """Return buoyancy operations."""
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            ops.add(Operation.create_SeqOp(func))
        return ops
```

### 4.4 Solver Collects Model Operations

```python
@Solver
class IncompressibleFluid(BaseModel):
    models: list[SolverModel] = []

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        """Collect operations from solver and all models."""
        ops = OperationCollection()

        # Solver's own operations
        for func in decorated_member_functions(self):
            ops.add(Operation.create_SeqOp(func))

        # Algorithm operations
        if self.algorithm_model:
            ops.add(self.algorithm_model.operations())

        # Model operations (extensions)
        for model in self.models:
            ops.add(model.operations())

        return ops
```

---

## 5. Model Extension System

### 5.1 Model Registry with Discovery

```python
# foamadapter/models/__init__.py

MODEL_REGISTRY: dict[str, type[SolverModel]] = {}

def register_model(name: str):
    """Decorator to register a model type."""
    def decorator(cls: type[SolverModel]) -> type[SolverModel]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str, **kwargs) -> SolverModel:
    """Create a model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](**kwargs)

def list_models() -> list[str]:
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())
```

### 5.2 Built-in Models

```python
# foamadapter/models/turbulence.py
@register_model("turbulence")
class TurbulenceModel(BaseModel):
    name: str = "turbulence"
    type: str = "kEpsilon"  # or "kOmegaSST", "LES", etc.
    ...

# foamadapter/models/buoyancy.py
@register_model("buoyancy")
class BuoyancyModel(BaseModel):
    name: str = "buoyancy"
    beta: float = 3e-3
    ...

# foamadapter/models/radiation.py
@register_model("radiation")
class RadiationModel(BaseModel):
    name: str = "radiation"
    type: str = "P1"
    ...

# foamadapter/models/species.py
@register_model("species")
class SpeciesTransportModel(BaseModel):
    name: str = "species"
    species: list[str] = ["CO2", "H2O"]
    ...
```

### 5.3 Configuration-Driven Composition

```yaml
# solver_config.yaml
solver:
  type: IncompressibleFluid
  argv: ["cavity"]
  algorithm: PIMPLE

  models:
    - type: turbulence
      model: kEpsilon

    - type: buoyancy
      beta: 3e-3
      TRef: 300.0

    - type: radiation
      model: P1
```

```python
# Load from config
config = load_config("solver_config.yaml")
solver = create_solver_from_config(config)
ctx = solver.initialize()
solver.run(ctx)
```

### 5.4 Plugin Models (External)

```python
# User's custom model package
# my_models/porous_zone.py

from foamadapter.models import register_model, SolverModel

@register_model("porous_zone")
class PorousZoneModel(BaseModel):
    """Custom porous zone model."""

    name: str = "porous_zone"
    zone_name: str = "porousZone"
    darcy_coefficient: float = 1e6
    forchheimer_coefficient: float = 100

    @Model.operation(depends_on=["momentum"])
    def porous_source(self, U, ctx) -> FieldUpdates:
        """Add porous zone resistance to momentum."""
        # Custom physics
        ...

# Usage
from my_models.porous_zone import PorousZoneModel

solver = (
    IncompressibleFluid(argv=["case"])
    .add_model(PorousZoneModel(zone_name="filter", darcy_coefficient=1e7))
)
```

---

## 6. Implementation Plan

### Phase 1: Context Creation in SETUP (Week 1)

```
□ Create ContextBuilder class
□ Update @Solver.setup signature to receive builder
□ Modify SolverInitializer._run_setup() to return Context
□ Add solver.initialize() convenience method
□ Remove duplicate state from IncompressibleFluid (self.p, self.U, etc.)
□ Deprecate create_context() method
□ Update all tests
```

### Phase 2: SolverModel Protocol (Week 2)

```
□ Define SolverModel protocol
□ Add models: list[SolverModel] to base solver
□ Add add_model() fluent method
□ Update get_models() to include composed models
□ Update operations() to collect from models
```

### Phase 3: Model Registry (Week 3)

```
□ Create model registry module
□ Add @register_model decorator
□ Implement get_model(), list_models()
□ Extract TurbulenceModel from solver
□ Create BuoyancyModel
□ Add configuration-driven composition
```

### Phase 4: Documentation & Examples (Week 4)

```
□ Document SolverModel protocol
□ Create example: custom porous zone model
□ Create example: multi-model solver
□ Add migration guide from old API
```

---

## Summary

### Before (Current)

```python
solver = IncompressibleFluid(argv=["cavity"])
initializer = SolverInitializer(solver)
initializer.initialize(mesh)
ctx = solver.create_context()  # Separate step!
solver.main_loop(ctx)

# State duplicated: solver.p AND ctx.fields["p"]
# Tightly coupled: can't add models
# Monolithic: all operations in solver class
```

### After (Proposed)

```python
solver = (
    IncompressibleFluid(argv=["cavity"])
    .add_model(TurbulenceModel(type="kEpsilon"))
    .add_model(BuoyancyModel(beta=3e-3))
)
ctx = solver.initialize()  # Returns Context!
solver.run(ctx)

# Single source of truth: only ctx.fields["p"]
# Composable: add any SolverModel
# Extensible: models contribute operations
```

### Key Benefits

| Aspect | Before | After |
|--------|--------|-------|
| State ownership | Duplicated | Context only |
| Context creation | Manual step | Part of initialize() |
| Model composition | Hardcoded | Pluggable |
| Operation sources | Solver only | Solver + models |
| Extensibility | Modify solver | Add models |
| Configuration | Code only | YAML + code |
| Testing | Complex mocking | Mock Context |
