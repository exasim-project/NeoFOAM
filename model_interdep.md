# Model Interdependency: Data Exchange Between Models

This document describes how models exchange data during initialization and proposes implementation strategies.

---

## Executive Summary

### Core Concept

Models in a CFD solver have two types of dependencies that occur at different times:

1. **Configuration Dependencies** (RESOLVE stage): One model changes another model's settings before anything is created
2. **Data Dependencies** (BUILD stage): One model's runtime objects need another model's runtime objects to exist first

The key insight is to **separate these concerns**:
- Use `AdaptableField` + `ModelRegistry` for configuration exchange
- Use `LazyInit` + DAG resolution for data dependencies

### Key Components

| Component | Purpose | Stage |
|-----------|---------|-------|
| `ModelRegistry` | Lookup models by name for configuration | RESOLVE |
| `AdaptableField` | Mark fields that other models can modify | RESOLVE |
| `LazyInit` | Deferred initialization with declared dependencies | BUILD |
| `DAGResolver` | Topologically sort and execute lazy inits | BUILD→EXECUTE |
| `Context` | Runtime access to all created objects | RUNTIME |

### Data Flow

```
LOAD          RESOLVE                    BUILD                 DAG              CONTEXT
─────────────────────────────────────────────────────────────────────────────────────────
Pydantic  →  Models modify each    →  Return LazyInit    →  Topo sort   →  Named access
models       other's AdaptableFields   with depends_on       & execute      to fields
```

### Minimal Example

```python
class VelocityModel(BaseModel):
    use_buoyancy: bool = AdaptableField(default=False)  # Can be modified

    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        deps = ["fields.p", "fields.nu"]
        if self.use_buoyancy:
            deps.append("sources.buoyancy")  # Conditional dependency
        return [LazyInit("operators.momentum", deps, create=...)]


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        registry.get("velocity").use_buoyancy = True  # Configure before build
```

---

## Problem Statement

In a CFD solver, models are interdependent:

- **VelocityModel** needs `nu` from **TransportModel**
- **PressureModel** needs `U` field from **VelocityModel**
- **BuoyancyModel** needs to tell **VelocityModel** to include buoyancy forces
- **TurbulenceModel** needs transport properties and modifies effective viscosity

The challenge: How do models communicate without tight coupling?

---

## Types of Model Interdependencies

### Type 1: Configuration Dependencies (RESOLVE Stage)

One model modifies another model's configuration **before** runtime objects are created.

```
BuoyancyModel → sets → VelocityModel.use_buoyancy = True
TurbulenceModel → sets → TransportModel.effective_viscosity_model = "kEpsilon"
```

**Timing**: During RESOLVE_DEPENDENCIES stage
**Mechanism**: Direct field modification via ModelRegistry

### Type 2: Data Dependencies (BUILD Stage)

One model's runtime objects depend on another model's runtime objects.

```
MomentumEquation → needs → [U field, p field, nu field]
PressureEquation → needs → [p field, U field, phi field]
```

**Timing**: During BUILD stage (lazy initialization)
**Mechanism**: DAG-based dependency resolution

### Type 3: Runtime Dependencies (Solve Stage)

Models exchange data during the simulation loop.

```
VelocityModel.solve() → updates → U field → read by → PressureModel.solve()
```

**Timing**: During solve loop
**Mechanism**: Shared Context with named fields

---

## RESOLVE Stage: Setter/Getter Design Proposals

The RESOLVE stage handles configuration dependencies where one model needs to modify another model's settings. Here are several design proposals from a setter/getter perspective.

### Proposal A: Direct Field Mutation (Current)

Models directly access and modify other models' fields.

```python
class VelocityModel(BaseModel):
    use_buoyancy: bool = AdaptableField(default=False)
    g: tuple = AdaptableField(default=(0, 0, -9.81))


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        # Direct mutation - simple but no validation
        velocity = registry.get("velocity")
        velocity.use_buoyancy = True
        velocity.g = self.gravity_vector
```

**Pros:**
- Simple and direct
- Familiar Python pattern
- Pydantic validates on assignment

**Cons:**
- No tracking of who changed what
- Hard to debug "why is this True?"
- No way to prevent conflicting changes

---

### Proposal B: Explicit Setters with Source Tracking

Models use explicit setter methods that track the source of changes.

```python
class VelocityModel(BaseModel):
    use_buoyancy: bool = AdaptableField(default=False)
    _change_log: list = []  # Track who changed what

    def set_use_buoyancy(self, value: bool, source: str):
        """Set with source tracking."""
        self._change_log.append({
            "field": "use_buoyancy",
            "old": self.use_buoyancy,
            "new": value,
            "source": source
        })
        self.use_buoyancy = value


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        velocity = registry.get("velocity")
        velocity.set_use_buoyancy(True, source="buoyancy_model")
```

**Pros:**
- Full audit trail
- Easy debugging ("buoyancy_model set use_buoyancy=True")
- Can detect conflicts

**Cons:**
- Verbose - need setter for each adaptable field
- Breaks Pydantic's clean field syntax

---

### Proposal C: Registry-Mediated Access

All access goes through the registry, which can enforce rules.

```python
class ModelRegistry:
    def get(self, name: str) -> "ModelProxy":
        """Return a proxy that mediates all access."""
        return ModelProxy(self._models[name], name, self)

    def set_field(self, model_name: str, field: str, value: Any, source: str):
        """Controlled field setting with validation."""
        model = self._models[model_name]

        # Check if field is adaptable
        if not self._is_adaptable(model, field):
            raise PermissionError(f"Field '{field}' is not adaptable")

        # Check for conflicts
        if (model_name, field) in self._pending_changes:
            prev_source = self._pending_changes[(model_name, field)]["source"]
            raise ConflictError(
                f"Field '{field}' already modified by '{prev_source}'"
            )

        # Record and apply
        self._pending_changes[(model_name, field)] = {
            "value": value, "source": source
        }
        setattr(model, field, value)


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        # All modifications go through registry
        registry.set_field("velocity", "use_buoyancy", True, source=self.name)
        registry.set_field("velocity", "g", self.gravity, source=self.name)
```

**Pros:**
- Centralized control and validation
- Conflict detection built-in
- Full audit trail

**Cons:**
- More verbose API
- Less Pythonic
- Registry becomes complex

---

### Proposal D: Request/Provide Pattern

Models declare what they need (getters) and what they provide (setters) explicitly.

```python
class VelocityModel(BaseModel):
    """Velocity model with explicit dependency interface."""

    # What this model provides to others
    class Provides:
        use_buoyancy: bool = AdaptableField(default=False)
        use_mrf: bool = AdaptableField(default=False)

    # Internal config (not adaptable)
    relax: float = 0.7
    provides: Provides = Provides()

    def accept(self, feature: str, config: dict, source: str):
        """Accept configuration from another model."""
        if feature == "buoyancy":
            self.provides.use_buoyancy = True
            self.provides.g = config.get("g", (0, 0, -9.81))


class BuoyancyModel(BaseModel):
    """Buoyancy model that requests velocity modifications."""

    # What this model needs from others
    class Requests:
        velocity: str = "buoyancy"  # Request "buoyancy" feature from velocity

    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        velocity = registry.get("velocity")
        velocity.accept("buoyancy", {"g": self.g, "T_ref": self.T_ref}, self.name)
```

**Pros:**
- Explicit interface contract
- Clear what's adaptable via `Provides`
- Feature-based grouping

**Cons:**
- More boilerplate
- Nested classes may be confusing
- Requires learning new pattern

---

### Proposal E: Event-based Configuration

Models emit configuration events that others can listen to.

```python
class ConfigEvent:
    """Event carrying configuration change."""
    def __init__(self, source: str, target: str, changes: dict):
        self.source = source
        self.target = target
        self.changes = changes


class VelocityModel(BaseModel):
    use_buoyancy: bool = AdaptableField(default=False)

    def on_config_event(self, event: ConfigEvent):
        """Handle configuration events from other models."""
        for field, value in event.changes.items():
            if hasattr(self, field) and self._is_adaptable(field):
                setattr(self, field, value)


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        # Emit event instead of direct modification
        registry.emit(ConfigEvent(
            source=self.name,
            target="velocity",
            changes={"use_buoyancy": True, "g": self.g}
        ))


class ModelRegistry:
    def emit(self, event: ConfigEvent):
        """Dispatch configuration event to target model."""
        target = self._models.get(event.target)
        if target and hasattr(target, "on_config_event"):
            target.on_config_event(event)
```

**Pros:**
- Decoupled - emitter doesn't know target's structure
- Target has full control over what to accept
- Easy to add logging/validation

**Cons:**
- Indirect - harder to trace
- Event objects add overhead
- Requires handler in every model

---

### Proposal F: Capability-based Access

Models expose capabilities that others can query and configure.

```python
from typing import Protocol

class BuoyancyCapable(Protocol):
    """Protocol for models that support buoyancy."""
    use_buoyancy: bool
    g: tuple[float, float, float]
    T_ref: float


class VelocityModel(BaseModel):
    """Velocity model implementing BuoyancyCapable."""
    use_buoyancy: bool = AdaptableField(default=False)
    g: tuple = AdaptableField(default=(0, 0, -9.81))
    T_ref: float = AdaptableField(default=300.0)


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        # Type-safe access via capability
        velocity = registry.get_capable("velocity", BuoyancyCapable)
        if velocity:
            velocity.use_buoyancy = True
            velocity.g = self.gravity
            velocity.T_ref = self.reference_temperature


class ModelRegistry:
    def get_capable(self, name: str, capability: type) -> Optional[Any]:
        """Get model only if it implements the capability."""
        model = self._models.get(name)
        if model and isinstance(model, capability):
            return model
        return None
```

**Pros:**
- Type-safe via Protocol
- Models explicitly opt-in to capabilities
- IDE autocomplete works
- Clear interface contracts

**Cons:**
- Requires defining Protocol for each capability
- Models must implement full protocol

---

### Comparison Matrix

| Aspect | A (Direct) | B (Setters) | C (Registry) | D (Request) | E (Events) | F (Capability) |
|--------|-----------|-------------|--------------|-------------|------------|----------------|
| **Simplicity** | ★★★★★ | ★★★ | ★★★ | ★★ | ★★ | ★★★★ |
| **Type Safety** | ★★★ | ★★★ | ★★★ | ★★★ | ★★ | ★★★★★ |
| **Traceability** | ★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★ |
| **Conflict Detection** | ★ | ★★★ | ★★★★★ | ★★★ | ★★★ | ★★ |
| **Decoupling** | ★★ | ★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★ |
| **IDE Support** | ★★★★★ | ★★★ | ★★ | ★★★ | ★★ | ★★★★★ |

---

### Recommendation

**Proposal F (Capability-based)** combined with **Proposal B (Source Tracking)** offers the best balance:

```python
from typing import Protocol

# Define capability protocols
class BuoyancyCapable(Protocol):
    use_buoyancy: bool
    g: tuple[float, float, float]


# Model implements capability
class VelocityModel(BaseModel):
    use_buoyancy: bool = AdaptableField(default=False)
    g: tuple = AdaptableField(default=(0, 0, -9.81))
    _configured_by: dict[str, str] = {}  # Track sources

    def configure(self, field: str, value: Any, source: str):
        """Configure with source tracking."""
        self._configured_by[field] = source
        setattr(self, field, value)


# Consumer uses type-safe access
class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        velocity = registry.get_capable("velocity", BuoyancyCapable)
        velocity.configure("use_buoyancy", True, source=self.name)
        velocity.configure("g", self.gravity, source=self.name)
```

This gives:
- Type-safe access via Protocol
- Source tracking for debugging
- Simple API
- IDE autocomplete

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOAD Stage                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Transport   │  │ Velocity    │  │ Pressure    │              │
│  │ Model       │  │ Model       │  │ Model       │              │
│  │ ─────────── │  │ ─────────── │  │ ─────────── │              │
│  │ nu: 1e-6    │  │ relax: 0.7  │  │ nCorr: 2    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         ↓                ↓                ↓                      │
│                    ModelRegistry                                 │
│         ┌────────────────┴────────────────┐                     │
└─────────┼─────────────────────────────────┼─────────────────────┘
          ↓                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RESOLVE Stage                                  │
│                                                                  │
│  registry.get("transport") ←──── VelocityModel reads nu         │
│  registry.get("velocity").use_buoyancy = True ←── BuoyancyModel │
│                                                                  │
│  Models can:                                                     │
│  • Read other models' fields                                     │
│  • Modify other models' AdaptableFields                         │
│  • Validate dependencies exist                                   │
└─────────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BUILD Stage                                   │
│                                                                  │
│  Each model returns LazyInit objects:                           │
│                                                                  │
│  TransportModel:                                                │
│    LazyInit("fields.nu", depends_on=["mesh"], ...)              │
│                                                                  │
│  VelocityModel:                                                 │
│    LazyInit("fields.U", depends_on=["mesh"], ...)               │
│    LazyInit("operators.momentum", depends_on=["fields.U",       │
│              "fields.p", "fields.nu"], ...)                     │
│                                                                  │
│  PressureModel:                                                 │
│    LazyInit("fields.p", depends_on=["mesh"], ...)               │
│    LazyInit("operators.pressure", depends_on=["fields.p",       │
│              "fields.U"], ...)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DAG Resolution                                  │
│                                                                  │
│  Topological sort of all LazyInit objects:                      │
│                                                                  │
│  1. mesh (no deps)                                              │
│  2. fields.nu (deps: mesh) ─────┐                               │
│  3. fields.U (deps: mesh) ──────┼─→ can run in parallel         │
│  4. fields.p (deps: mesh) ──────┘                               │
│  5. operators.momentum (deps: U, p, nu)                         │
│  6. operators.pressure (deps: p, U)                             │
│  7. piso_loop (deps: momentum, pressure)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Context                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ fields: {"nu": νField, "U": UField, "p": pField, ...}   │    │
│  │ operators: {"momentum": momEq, "pressure": pEq}         │    │
│  │ mesh: fvMesh                                            │    │
│  │ runtime: Time                                           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Proposal

### 1. ModelRegistry with Type-Safe Access

```python
from typing import TypeVar, Type, Optional
from dataclasses import dataclass

T = TypeVar('T')

class ModelRegistry:
    """Registry for inter-model communication during RESOLVE stage."""

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._type_index: dict[type, list[str]] = {}

    def register(self, name: str, model: Any) -> None:
        """Register a model by name."""
        self._models[name] = model

        # Index by type for type-safe retrieval
        model_type = type(model)
        if model_type not in self._type_index:
            self._type_index[model_type] = []
        self._type_index[model_type].append(name)

    def get(self, name: str) -> Any:
        """Get model by name (untyped)."""
        return self._models.get(name)

    def get_typed(self, name: str, expected_type: Type[T]) -> T:
        """Get model by name with type checking."""
        model = self._models.get(name)
        if model is None:
            raise KeyError(f"Model '{name}' not found in registry")
        if not isinstance(model, expected_type):
            raise TypeError(
                f"Model '{name}' is {type(model).__name__}, "
                f"expected {expected_type.__name__}"
            )
        return model

    def get_all_of_type(self, model_type: Type[T]) -> list[T]:
        """Get all models of a specific type."""
        names = self._type_index.get(model_type, [])
        return [self._models[name] for name in names]

    def require(self, name: str) -> Any:
        """Get model or raise if not found."""
        model = self._models.get(name)
        if model is None:
            raise DependencyError(f"Required model '{name}' not found")
        return model
```

### 2. AdaptableField for Configuration Exchange

```python
from pydantic import Field

def AdaptableField(**kwargs):
    """
    Mark a field as modifiable by other models during RESOLVE stage.

    Other models can modify these fields to configure behavior.
    """
    json_schema_extra = kwargs.get("json_schema_extra", {}) or {}
    json_schema_extra["adaptable"] = True
    kwargs["json_schema_extra"] = json_schema_extra
    return Field(**kwargs)


class VelocityModel(BaseModel):
    """Velocity field - can be adapted by other models."""
    name: str = "velocity"

    # These can be modified by other models
    use_buoyancy: bool = AdaptableField(default=False)
    use_mrf: bool = AdaptableField(default=False)
    body_forces: list[str] = AdaptableField(default_factory=list)

    # These are read-only after LOAD
    relax: float = Field(default=0.7)


class BuoyancyModel(BaseModel):
    """Buoyancy model - modifies velocity model."""
    name: str = "buoyancy"
    enabled: bool = True

    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        if self.enabled:
            velocity = registry.get_typed("velocity", VelocityModel)
            velocity.use_buoyancy = True
            velocity.body_forces.append("buoyancy")


class MRFModel(BaseModel):
    """Moving Reference Frame - modifies velocity model."""
    name: str = "mrf"
    zones: list[str] = []

    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        if self.zones:
            velocity = registry.get_typed("velocity", VelocityModel)
            velocity.use_mrf = True
```

### 3. Dependency Injection for BUILD Stage

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class LazyInit:
    """Deferred initialization with explicit dependencies."""
    name: str
    depends_on: list[str]
    initializer: Callable[..., Any]

    def execute(self, resolved: dict[str, Any]) -> Any:
        """Execute with resolved dependencies injected."""
        # Filter to only the dependencies we need
        deps = {k: v for k, v in resolved.items() if k in self.depends_on}
        return self.initializer(**deps)


# Helper functions for common patterns
def field(name: str, *, depends_on: list[str] = None, create: Callable) -> LazyInit:
    """Create a LazyInit for a field."""
    return LazyInit(
        name=f"fields.{name}",
        depends_on=depends_on or ["mesh"],
        initializer=create
    )


def operator(name: str, *, depends_on: list[str], create: Callable) -> LazyInit:
    """Create a LazyInit for an operator."""
    return LazyInit(
        name=f"operators.{name}",
        depends_on=depends_on,
        initializer=create
    )


# Usage in model
class VelocityModel(BaseModel):
    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            field("U", create=lambda mesh: volVectorField(mesh, "U")),

            field("phi",
                  depends_on=["mesh", "fields.U"],
                  create=lambda mesh, U: createPhi(mesh, U)),

            operator("momentum",
                     depends_on=["fields.U", "fields.p", "fields.nu"],
                     create=lambda U, p, nu: MomentumEquation(U, p, nu,
                         include_buoyancy=self.use_buoyancy))
        ]
```

### 4. DAG Resolver

```python
from collections import defaultdict
from typing import Any

class DependencyError(Exception):
    """Raised when dependencies cannot be resolved."""
    pass


class DAGResolver:
    """Resolves initialization order using topological sort."""

    def __init__(self, lazy_inits: list[LazyInit]):
        self.lazy_inits = {li.name: li for li in lazy_inits}
        self._validate()

    def _validate(self):
        """Check for missing and circular dependencies."""
        all_names = set(self.lazy_inits.keys())

        # Check for missing dependencies
        for li in self.lazy_inits.values():
            missing = set(li.depends_on) - all_names
            if missing:
                raise DependencyError(
                    f"'{li.name}' depends on missing: {missing}"
                )

        # Check for cycles (DFS)
        self._detect_cycles()

    def _detect_cycles(self):
        """Detect circular dependencies using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.lazy_inits}

        def dfs(name: str, path: list[str]):
            color[name] = GRAY
            path.append(name)

            for dep in self.lazy_inits[name].depends_on:
                if color[dep] == GRAY:
                    # Found cycle
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    raise DependencyError(
                        f"Circular dependency: {' → '.join(cycle)}"
                    )
                if color[dep] == WHITE:
                    dfs(dep, path)

            path.pop()
            color[name] = BLACK

        for name in self.lazy_inits:
            if color[name] == WHITE:
                dfs(name, [])

    def resolve(self) -> list[LazyInit]:
        """Return LazyInits in dependency order (Kahn's algorithm)."""
        # Count incoming edges
        in_degree = {name: 0 for name in self.lazy_inits}
        for li in self.lazy_inits.values():
            for dep in li.depends_on:
                # dep has an outgoing edge to li.name
                pass  # We count reverse

        # Build adjacency list (dependency → dependents)
        dependents = defaultdict(list)
        for li in self.lazy_inits.values():
            for dep in li.depends_on:
                dependents[dep].append(li.name)
            in_degree[li.name] = len(li.depends_on)

        # Start with nodes that have no dependencies
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            name = queue.pop(0)
            result.append(self.lazy_inits[name])

            # Reduce in-degree for dependents
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    def execute_all(self) -> dict[str, Any]:
        """Execute all initializers in order, returning results."""
        order = self.resolve()
        resolved = {}

        for li in order:
            result = li.execute(resolved)
            resolved[li.name] = result

        return resolved
```

### 5. Context for Runtime Data Exchange

```python
from typing import Any, Optional

class Context:
    """
    Runtime container for simulation state.

    Provides typed access to fields, operators, and infrastructure.
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, name: str) -> Any:
        """Get any named object."""
        if name not in self._data:
            raise KeyError(f"'{name}' not found in context")
        return self._data[name]

    def field(self, name: str) -> Any:
        """Get a field by name."""
        return self.get(f"fields.{name}")

    def operator(self, name: str) -> Any:
        """Get an operator by name."""
        return self.get(f"operators.{name}")

    @property
    def mesh(self) -> Any:
        """Get the mesh."""
        return self.get("mesh")

    @property
    def runtime(self) -> Any:
        """Get the time controller."""
        return self.get("runtime")

    def all_fields(self) -> dict[str, Any]:
        """Get all fields."""
        return {
            k.replace("fields.", ""): v
            for k, v in self._data.items()
            if k.startswith("fields.")
        }

    def all_operators(self) -> dict[str, Any]:
        """Get all operators."""
        return {
            k.replace("operators.", ""): v
            for k, v in self._data.items()
            if k.startswith("operators.")
        }
```

---

## Complete Example: Buoyancy Coupling

```python
from pydantic import BaseModel, Field
from foamadapter.framework import Model, Solver, AdaptableField, LazyInit

# ============================================================================
# Transport Model
# ============================================================================

class TransportModel(BaseModel):
    """Transport properties."""
    name: str = "transport"
    nu: float = None
    beta: float = None  # Thermal expansion coefficient

    @Model.load
    def load(self):
        props = read_dict("constant/transportProperties")
        self.nu = props["nu"]
        self.beta = props.get("beta", 0.0)

    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            LazyInit("fields.nu", ["mesh"],
                     lambda mesh: volScalarField(mesh, self.nu)),
            LazyInit("fields.beta", ["mesh"],
                     lambda mesh: volScalarField(mesh, self.beta))
        ]


# ============================================================================
# Velocity Model (can be adapted)
# ============================================================================

class VelocityModel(BaseModel):
    """Velocity field with configurable physics."""
    name: str = "velocity"

    # Adaptable by other models
    use_buoyancy: bool = AdaptableField(default=False)
    g: tuple[float, float, float] = AdaptableField(default=(0, 0, -9.81))
    T_ref: float = AdaptableField(default=300.0)

    # Fixed after load
    relax: float = Field(default=0.7)

    @Model.load
    def load(self):
        self.relax = read_dict("system/fvSolution")["relaxationFactors"]["U"]

    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        inits = [
            LazyInit("fields.U", ["mesh"],
                     lambda mesh: volVectorField(mesh, "U"))
        ]

        # Conditionally add buoyancy source
        if self.use_buoyancy:
            inits.append(
                LazyInit("sources.buoyancy",
                         ["fields.T", "fields.beta", "mesh"],
                         lambda T, beta, mesh: BuoyancySource(
                             mesh, T, beta, self.g, self.T_ref
                         ))
            )

        # Momentum equation depends on buoyancy if enabled
        mom_deps = ["fields.U", "fields.p", "fields.nu"]
        if self.use_buoyancy:
            mom_deps.append("sources.buoyancy")

        inits.append(
            LazyInit("operators.momentum", mom_deps,
                     lambda **deps: MomentumEquation(
                         deps["fields.U"],
                         deps["fields.p"],
                         deps["fields.nu"],
                         buoyancy=deps.get("sources.buoyancy")
                     ))
        )

        return inits


# ============================================================================
# Temperature Model
# ============================================================================

class TemperatureModel(BaseModel):
    """Temperature field for thermal simulations."""
    name: str = "temperature"

    @Model.load
    def load(self):
        pass  # Read from 0/T

    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            LazyInit("fields.T", ["mesh"],
                     lambda mesh: volScalarField(mesh, "T"))
        ]


# ============================================================================
# Buoyancy Model (adapts VelocityModel)
# ============================================================================

class BuoyancyModel(BaseModel):
    """Buoyancy physics - modifies velocity model."""
    name: str = "buoyancy"
    enabled: bool = True
    g: tuple[float, float, float] = (0, 0, -9.81)
    T_ref: float = 300.0

    @Model.load
    def load(self):
        g_dict = read_dict("constant/g")
        self.g = tuple(g_dict["value"])

    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        """Configure velocity model for buoyancy."""
        if not self.enabled:
            return

        # Require temperature model
        if not registry.contains("temperature"):
            raise DependencyError(
                "BuoyancyModel requires TemperatureModel"
            )

        # Configure velocity model
        velocity = registry.require("velocity")
        velocity.use_buoyancy = True
        velocity.g = self.g
        velocity.T_ref = self.T_ref

        # Also need thermal expansion from transport
        transport = registry.require("transport")
        if transport.beta == 0.0:
            raise DependencyError(
                "BuoyancyModel requires beta in transportProperties"
            )


# ============================================================================
# Solver
# ============================================================================

class BoussinesqSolver(BaseModel):
    """Natural convection solver with Boussinesq approximation."""

    transport: TransportModel = TransportModel()
    velocity: VelocityModel = VelocityModel()
    temperature: TemperatureModel = TemperatureModel()
    buoyancy: BuoyancyModel = BuoyancyModel()

    @Solver.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            LazyInit("mesh", [], lambda: mesh),
            LazyInit("runtime", [],
                     lambda: Time(read_dict("system/controlDict"))),
            LazyInit("solver.loop",
                     ["operators.momentum", "operators.energy", "runtime"],
                     lambda **deps: SolveLoop(deps))
        ]


# ============================================================================
# Usage
# ============================================================================

solver = BoussinesqSolver()
initializer = SolverInitializer(solver)
ctx = initializer.initialize(mesh)

# Execution order (determined by DAG):
# 1. mesh
# 2. runtime
# 3. fields.nu, fields.beta, fields.U, fields.T (parallel - no inter-deps)
# 4. sources.buoyancy (needs T, beta)
# 5. operators.momentum (needs U, p, nu, buoyancy)
# 6. operators.energy
# 7. solver.loop
```

---

## Inter-Model Communication Patterns

### Pattern 1: Feature Flag

One model enables/disables features in another.

```python
class TurbulenceModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        # Enable turbulent transport
        transport = registry.get("transport")
        transport.use_turbulent_viscosity = True
```

### Pattern 2: Parameter Injection

One model provides parameters to another.

```python
class WallFunctionModel(BaseModel):
    y_plus_target: float = 30.0

    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        turbulence = registry.get("turbulence")
        turbulence.wall_treatment = "wallFunction"
        turbulence.y_plus = self.y_plus_target
```

### Pattern 3: Dependency Validation

Ensure required models exist.

```python
class CombustionModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        # Require specific models
        registry.require("species")
        registry.require("thermodynamics")
        registry.require("turbulence")
```

### Pattern 4: Conditional Dependencies

Add dependencies based on configuration.

```python
class VelocityModel(BaseModel):
    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        deps = ["fields.U", "fields.p"]

        if self.use_buoyancy:
            deps.append("sources.buoyancy")
        if self.use_mrf:
            deps.append("mrf.zone")

        return [
            LazyInit("operators.momentum", deps,
                     create=self._create_momentum)
        ]
```

### Pattern 5: Multiple Contributions

Multiple models contribute to a shared resource.

```python
class SourceTermCollector:
    """Collects source terms from multiple models."""
    def __init__(self):
        self.sources = []

    def add(self, name: str, source: Any):
        self.sources.append((name, source))


class BuoyancyModel(BaseModel):
    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            LazyInit("sources.buoyancy", ["fields.T"],
                     create=lambda T: BuoyancySource(T, self.g))
        ]


class PorousZoneModel(BaseModel):
    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            LazyInit("sources.porous", ["fields.U"],
                     create=lambda U: DarcySource(U, self.D))
        ]


class MomentumModel(BaseModel):
    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        # Gather all source terms
        return [
            LazyInit("operators.momentum",
                     depends_on=["fields.U", "fields.p",
                                 "sources.buoyancy", "sources.porous"],
                     create=lambda U, p, buoyancy, porous:
                         MomentumEquation(U, p, sources=[buoyancy, porous]))
        ]
```

---

## Summary

| Stage | Purpose | Data Exchange Mechanism |
|-------|---------|------------------------|
| **LOAD** | Init Pydantic models | None (isolated) |
| **RESOLVE** | Configure models | ModelRegistry + AdaptableField |
| **BUILD** | Create lazy inits | LazyInit with explicit depends_on |
| **DAG** | Order execution | Topological sort |
| **EXECUTE** | Run initializers | Dependency injection |
| **RUNTIME** | Solve loop | Context with named access |

This design provides:
- **Loose coupling**: Models only know names, not types
- **Explicit dependencies**: No hidden side effects
- **Type safety**: Optional typed access via registry
- **Testability**: Easy to mock dependencies
- **Parallelism**: DAG enables parallel initialization
