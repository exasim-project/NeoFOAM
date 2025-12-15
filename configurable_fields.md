# Configurable Fields: Type-Based Inter-Model Configuration

A field's **type** indicates whether other models can modify it.

---

## TL;DR - Recommended Approach

After reviewing all proposals, here is the recommended design for inter-model configuration:

### Design Decision Summary

| Aspect | Recommendation | Rationale |
|--------|----------------|-----------|
| **Configurable fields** | `Configurable[T]` type annotation | Type shows intent, Pydantic compatible, IDE support |
| **Model lookup** | `ConfigContext` | Clear name, supports intra/inter-region via naming |
| **Method signature** | `resolve(self, config: ConfigContext)` | Uniform interface for all models |
| **Cross-region access** | `config.get("fluid.temperature")` | Dot notation for region.model |

### Naming Convention for Model Access

```python
# Intra-region (same region) - just model name
config.get("velocity")           # → VelocityModel in current region
config.get("transport")          # → TransportModel in current region

# Inter-region (cross-region) - region.model notation
config.get("fluid.temperature")  # → TemperatureModel in fluid region
config.get("solid.conductivity") # → ConductivityModel in solid region

# Explicit current region (optional)
config.get("default.velocity")   # → Same as config.get("velocity")
```

### Minimal Example

```python
from typing import Annotated
from pydantic import BaseModel

# Type marker for configurable fields
Configurable = Annotated

class VelocityModel(BaseModel):
    """Velocity model with configurable buoyancy support."""
    
    # Configurable by other models (type shows intent)
    use_buoyancy: Configurable[bool, "configurable"] = False
    g: Configurable[tuple, "configurable"] = (0, 0, -9.81)
    
    # Not configurable (regular field)
    relax: float = 0.7
    
    @Model.resolve_dependencies
    def resolve(self, config: ConfigContext):
        # Access sibling models (same region)
        transport = config.get("transport")
        self.nu = transport.nu


class BuoyancyModel(BaseModel):
    """Buoyancy model - configures VelocityModel."""
    enabled: bool = True
    gravity: tuple = (0, 0, -9.81)
    
    @Model.resolve_dependencies
    def resolve(self, config: ConfigContext):
        if self.enabled:
            velocity = config.get("velocity")
            velocity.use_buoyancy = True  # Set Configurable field
            velocity.g = self.gravity


class CHTCouplingModel(BaseModel):
    """Cross-region coupling using dot notation."""
    
    @Model.resolve_dependencies
    def resolve(self, config: ConfigContext):
        # Access models in different regions via naming convention
        fluid_T = config.get("fluid.temperature")
        solid_T = config.get("solid.temperature")
        
        # Configure coupling between regions
        fluid_T.coupled_to = solid_T
        solid_T.coupled_to = fluid_T
```

### ConfigContext API

```python
class ConfigContext:
    """Context for inter-model configuration exchange."""
    
    def __init__(self, current_region: str, solver: "Solver"):
        self.current_region = current_region
        self.solver = solver
    
    def get(self, path: str) -> Any:
        """
        Get a model by path.
        
        - "model_name" → model in current region
        - "region.model_name" → model in specified region
        """
        if "." in path:
            region, model = path.split(".", 1)
        else:
            region = self.current_region
            model = path
        
        return self.solver.regions[region].models[model]
    
    @property
    def mesh(self) -> Any:
        """Current region's mesh."""
        return self.solver.regions[self.current_region].mesh
    
    @property  
    def region(self) -> str:
        """Current region name."""
        return self.current_region
```

### Why This Design?

1. **`Configurable[T]`** - The type annotation documents which fields can be modified by other models. No runtime wrapper needed, just metadata.

2. **`ConfigContext`** - Clear name indicating purpose (configuration exchange). Single API for both intra and inter-region access.

3. **Dot notation** - Intuitive `region.model` syntax for cross-region access. No need for separate `get_region()` method.

4. **Uniform API** - Same `config.get()` for all cases. The presence of a dot determines scope.

---

## Current Implementation

The codebase currently uses `ModelRegistry` + `AdaptableField`:

```python
# Model with adaptable field
class PressureAlgorithmSimple(BaseModel):
    use_buoyancy: bool = AdaptableField(default=False)

# Model that modifies it
class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def adapt_pressure(self, registry: ModelRegistry):
        pressure = registry.get("adaptive_pressure")
        if pressure and self.enabled:
            pressure.use_buoyancy = True  # Direct mutation
```

**What gets exchanged:**
- Configuration flags (`use_buoyancy = True`)
- Model references (`self.transport_ref = registry.get("transport")`)
- Algorithm choices, parameters

**Problem with name "ModelRegistry":** It's really about **configuration exchange**, not model registration. The registry is just a lookup mechanism.

---

## Better Naming

| Current | Better Names | Rationale |
|---------|--------------|-----------|
| `ModelRegistry` | `SolverContext` | It's the context for inter-model communication |
| | `ConfigContext` | Focuses on configuration exchange |
| | Just `solver` | The solver itself holds all models |

Since the solver already contains all models as attributes, the simplest approach is to pass the **solver** directly:

```python
# Current
def resolve(self, registry: ModelRegistry):
    pressure = registry.get("pressure")

# Simpler - solver IS the context
def resolve(self, solver):
    pressure = solver.pressure  # Direct attribute access, IDE autocomplete works
```

---

## Multi-Region / Multi-Physics Considerations

The simple "pass solver" approach works for single-region solvers, but **multi-physics simulations** require a more flexible architecture:

### Scenarios

1. **Conjugate Heat Transfer (CHT)**: Fluid region + Solid region, coupled at interface
2. **Multi-Region Fluid**: Multiple fluid domains with different meshes
3. **Fluid-Structure Interaction (FSI)**: Fluid solver + Structural solver, two-way coupling

### The Problem

```python
# Single-region: solver holds all models
class IcoFoamSolver:
    velocity: VelocityModel
    pressure: PressureModel

# Multi-region: need hierarchy
class CHTSolver:
    fluid: FluidRegion      # Has its own velocity, pressure, transport
    solid: SolidRegion      # Has its own temperature, thermal properties
    coupling: Interface     # Couples fluid.T ↔ solid.T at boundary
```

Models now need to access:
- Sibling models in the **same region**
- Models in **other regions** (for coupling)
- **Global** solver configuration

### Proposed Architecture for Multi-Region

```
┌─────────────────────────────────────────────────────────────────┐
│                        MultiRegionSolver                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  FluidRegion    │  │  SolidRegion    │  │  CouplingModels │  │
│  │  ────────────── │  │  ────────────── │  │  ────────────── │  │
│  │  mesh: fvMesh   │  │  mesh: fvMesh   │  │  cht_interface  │  │
│  │  velocity       │  │  temperature    │  │  radiation      │  │
│  │  pressure       │  │  conductivity   │  │                 │  │
│  │  transport      │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Proposal: Hierarchical Context

```python
class ConfigContext:
    """Context for a single region - provides access to local models."""
    def __init__(self, region_name: str, solver: "MultiRegionSolver"):
        self.region_name = region_name
        self.solver = solver
        self._models: dict[str, Any] = {}
    
    def get(self, name: str) -> Any:
        """Get model in this region."""
        return self._models.get(name)
    
    def get_region(self, region_name: str) -> "ConfigContext":
        """Get another region's context (for coupling)."""
        return self.solver.get_region(region_name)
    
    @property
    def mesh(self):
        """This region's mesh."""
        return self._mesh


class MultiRegionSolver(BaseModel):
    """Solver managing multiple regions."""
    regions: dict[str, ConfigContext] = {}
    coupling: list[CouplingModel] = []
    
    def get_region(self, name: str) -> ConfigContext:
        return self.regions[name]
```

### Usage in Multi-Region

```python
class FluidTemperatureModel(BaseModel):
    """Temperature in fluid region - couples with solid."""
    
    @Model.resolve_dependencies
    def resolve(self, context: ConfigContext):
        # Access sibling in same region
        transport = context.get("transport")
        self.Pr = transport.Pr
        
        # Access model in another region (for coupling info)
        solid = context.get_region("solid")
        solid_temp = solid.get("temperature")
        
        # Check if we need to couple
        self.coupled_regions = ["solid"] if solid_temp else []


class CHTCouplingModel(BaseModel):
    """Couples temperature between fluid and solid regions."""
    
    @Model.resolve_dependencies
    def resolve(self, solver: MultiRegionSolver):
        # Access both regions
        fluid = solver.get_region("fluid")
        solid = solver.get_region("solid")
        
        # Configure coupling
        fluid_T = fluid.get("temperature")
        solid_T = solid.get("temperature")
        
        fluid_T.add_boundary_coupling("interface", solid_T)
        solid_T.add_boundary_coupling("interface", fluid_T)
```

### Context Naming for Multi-Region

| Scope | Name | What It Provides |
|-------|------|------------------|
| Single Region | `ConfigContext` | Models in current region |
| Cross Region | `solver.get_region(name)` | Access to other regions |
| Global | `solver` | All regions, global config |

### Resolve Method Signatures

```python
# Region model: receives ConfigContext
@Model.resolve_dependencies
def resolve(self, context: ConfigContext):
    sibling = context.get("transport")

# Coupling model: receives full solver
@Coupling.resolve_dependencies  
def resolve(self, solver: MultiRegionSolver):
    region_a = solver.get_region("fluid")
    region_b = solver.get_region("solid")
```

---

## Recommendation

Based on multi-region requirements, here is the recommended design:

### 1. Use `Configurable[T]` Type Annotation

```python
from typing import Annotated

# Type indicates the field can be modified by other models
Configurable = Annotated

class VelocityModel(BaseModel):
    use_buoyancy: Configurable[bool, "configurable"] = False  # Can be set by others
    relax: float = 0.7  # Cannot be modified by others
```

### 2. Use `ConfigContext` Instead of `ModelRegistry`

```python
class ConfigContext:
    """Context for inter-model communication within a region."""
    region_name: str
    solver: "Solver"
    
    def get(self, name: str) -> Any:
        """Get model in this region."""
    
    def get_region(self, name: str) -> "ConfigContext":
        """Get another region (for coupling)."""
    
    @property
    def mesh(self) -> Any:
        """This region's mesh."""
```

### 3. Pass `ConfigContext` to Resolve Methods

```python
class BuoyancyModel(BaseModel):
    enabled: bool = True
    g: tuple = (0, 0, -9.81)
    
    @Model.resolve_dependencies
    def resolve(self, context: ConfigContext):
        # Get sibling model and configure it
        velocity = context.get("velocity")
        velocity.use_buoyancy = True  # Set Configurable field
        velocity.g = self.g
```

### 4. Single-Region is Just One Region

```python
# Single-region solver
class IcoFoamSolver(BaseModel):
    regions: dict[str, Region] = {"default": Region(...)}

# Access is the same
context.get("velocity")  # Gets from "default" region
```

### Summary

| Component | Name | Purpose |
|-----------|------|---------|
| Field type | `Configurable[T]` | Marks field as settable by other models |
| Lookup mechanism | `ConfigContext` | Provides model access within/across regions |
| Method argument | `context: ConfigContext` | Passed to `@Model.resolve_dependencies` |
| Cross-region | `context.get_region("solid")` | Access other region's models |

This design:
- ✅ Works for single-region (one "default" region)
- ✅ Scales to multi-region/multi-physics
- ✅ Type annotation shows intent (`Configurable[T]`)
- ✅ Clear naming (`ConfigContext` not `ModelRegistry`)

---

## Core Concept

```python
from foamadapter.framework import Configurable

class VelocityModel(BaseModel):
    # Regular field - only this model can change it
    relax: float = 0.7
    
    # Configurable field - other models can set this
    use_buoyancy: Configurable[bool] = False
    body_forces: Configurable[list[str]] = []
```

The `Configurable[T]` type wrapper signals: "other models may modify this field during RESOLVE stage."

---

## Proposal A: Generic Type Wrapper

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Configurable(Generic[T]):
    """A field that can be configured by other models."""
    
    def __init__(self, default: T):
        self.value = default
        self.configured_by: str | None = None
    
    def set(self, value: T, source: str):
        self.value = value
        self.configured_by = source
    
    def get(self) -> T:
        return self.value


# Usage in model definition
class VelocityModel(BaseModel):
    use_buoyancy: Configurable[bool] = Configurable(False)
    g: Configurable[tuple] = Configurable((0, 0, -9.81))
    
    # Access the value
    def momentum_equation(self):
        if self.use_buoyancy.get():
            add_buoyancy_term(self.g.get())


# Usage in RESOLVE stage
class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        velocity = registry.get("velocity")
        velocity.use_buoyancy.set(True, source=self.name)
        velocity.g.set(self.gravity, source=self.name)
```

---

## Proposal B: Annotated Type with Pydantic

```python
from typing import Annotated
from pydantic import BaseModel

# Marker type
class ConfigurableMarker:
    """Marker indicating field is configurable by other models."""
    pass

Configurable = Annotated[T, ConfigurableMarker()]


class VelocityModel(BaseModel):
    # Type annotation shows it's configurable
    use_buoyancy: Configurable[bool] = False
    g: Configurable[tuple[float, float, float]] = (0, 0, -9.81)
    
    # Regular field - not configurable
    relax: float = 0.7


# Framework validates on set
class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        velocity = registry.get("velocity")
        
        # Works - field is Configurable
        registry.configure(velocity, "use_buoyancy", True, source=self.name)
        
        # Raises error - field is not Configurable
        registry.configure(velocity, "relax", 0.5, source=self.name)  # Error!
```

---

## Proposal C: Descriptor-Based (Cleanest Syntax)

```python
class Configurable:
    """Descriptor for configurable fields."""
    
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_cfg_{name}"
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, self.default)
    
    def __set__(self, obj, value):
        # Track who set it
        if isinstance(value, tuple) and len(value) == 2:
            val, source = value
            obj._configured_by[self.name] = source
            setattr(obj, self.private_name, val)
        else:
            setattr(obj, self.private_name, value)


class VelocityModel(BaseModel):
    use_buoyancy = Configurable(default=False)
    g = Configurable(default=(0, 0, -9.81))
    
    relax: float = 0.7  # Regular field


# Clean usage
class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies  
    def resolve(self, registry: ModelRegistry):
        velocity = registry.get("velocity")
        
        # Set with source tracking
        velocity.use_buoyancy = (True, self.name)
        velocity.g = (self.gravity, self.name)
```

---

## Proposal D: Protocol + Mixin (Type-Safe)

```python
from typing import Protocol

class Configurable(Protocol[T]):
    """Protocol for configurable values."""
    value: T
    configured_by: str | None
    
    def configure(self, value: T, source: str) -> None: ...


class ConfigurableValue(Generic[T]):
    """Implementation of Configurable."""
    def __init__(self, default: T):
        self.value = default
        self.configured_by = None
    
    def configure(self, value: T, source: str):
        self.value = value
        self.configured_by = source


# Shorthand
def configurable(default: T) -> Configurable[T]:
    return ConfigurableValue(default)


class VelocityModel(BaseModel):
    use_buoyancy: Configurable[bool] = configurable(False)
    g: Configurable[Vector] = configurable(Vector(0, 0, -9.81))


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        velocity = registry.get("velocity")
        velocity.use_buoyancy.configure(True, source=self.name)
```

---

## Comparison

| Proposal | Syntax | Type Safety | Pydantic Compatible | IDE Support |
|----------|--------|-------------|---------------------|-------------|
| A (Generic) | `Configurable(False)` | ★★★★ | ★★★ | ★★★★ |
| B (Annotated) | `Configurable[bool] = False` | ★★★★★ | ★★★★★ | ★★★★★ |
| C (Descriptor) | `Configurable(default=False)` | ★★★ | ★★ | ★★★ |
| D (Protocol) | `configurable(False)` | ★★★★★ | ★★★ | ★★★★★ |

---

## Recommendation: Proposal B

Use `Annotated` type with Pydantic - cleanest syntax, full type safety:

```python
from typing import Annotated

# Simple definition
Configurable = Annotated

class VelocityModel(BaseModel):
    use_buoyancy: Configurable[bool, "buoyancy"] = False  # Tag with feature
    relax: float = 0.7  # Not configurable
```

The type annotation itself documents the intent. Framework validates at runtime.

---

## Alternatives to ModelRegistry

The `ModelRegistry` is not the only way for models to find each other. Here are alternatives:

### Alternative 1: Solver as Mediator

Models don't access each other directly. The solver orchestrates all configuration.

```python
class IcoFoamSolver(BaseModel):
    transport: TransportModel
    velocity: VelocityModel
    buoyancy: BuoyancyModel
    
    @Solver.resolve_dependencies
    def resolve(self):
        # Solver knows all models - does the wiring
        if self.buoyancy.enabled:
            self.velocity.use_buoyancy.configure(True, source="solver")
            self.velocity.g.configure(self.buoyancy.g, source="solver")
```

**Pros:** Centralized logic, no hidden dependencies
**Cons:** Solver becomes complex, models can't self-configure

---

### Alternative 2: Dependency Injection via Constructor

Models declare dependencies as constructor parameters.

```python
class VelocityModel(BaseModel):
    use_buoyancy: Configurable[bool] = False


class BuoyancyModel(BaseModel):
    # Explicit dependency - injected at construction
    velocity: VelocityModel
    
    def __init__(self, velocity: VelocityModel, **kwargs):
        super().__init__(**kwargs)
        self.velocity = velocity
    
    @Model.resolve_dependencies
    def resolve(self):
        # Direct access - no registry needed
        self.velocity.use_buoyancy.configure(True, source=self.name)


# Solver wires them together
class IcoFoamSolver(BaseModel):
    def __init__(self):
        self.velocity = VelocityModel()
        self.buoyancy = BuoyancyModel(velocity=self.velocity)
```

**Pros:** Explicit dependencies, testable, type-safe
**Cons:** Manual wiring, circular dependencies problematic

---

### Alternative 3: Trait/Capability Queries

Models query for capabilities, not specific models.

```python
from typing import Protocol

class BuoyancyCapable(Protocol):
    """Any model that can include buoyancy."""
    use_buoyancy: Configurable[bool]
    g: Configurable[tuple]


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, solver):  # Receive solver, not registry
        # Find all models that support buoyancy
        for model in solver.get_models():
            if isinstance(model, BuoyancyCapable):
                model.use_buoyancy.configure(True, source=self.name)
```

**Pros:** Loose coupling, works with any supporting model
**Cons:** Less explicit, harder to trace

---

### Alternative 4: Event Bus / Pub-Sub

Models publish configuration requests, others subscribe.

```python
class ConfigBus:
    """Central event bus for configuration."""
    _handlers: dict[str, list[Callable]] = {}
    
    @classmethod
    def subscribe(cls, topic: str, handler: Callable):
        cls._handlers.setdefault(topic, []).append(handler)
    
    @classmethod
    def publish(cls, topic: str, **kwargs):
        for handler in cls._handlers.get(topic, []):
            handler(**kwargs)


class VelocityModel(BaseModel):
    use_buoyancy: Configurable[bool] = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Subscribe to buoyancy configuration
        ConfigBus.subscribe("enable_buoyancy", self._on_buoyancy)
    
    def _on_buoyancy(self, g, source):
        self.use_buoyancy.configure(True, source=source)
        self.g.configure(g, source=source)


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self):
        # Publish - don't care who listens
        ConfigBus.publish("enable_buoyancy", g=self.gravity, source=self.name)
```

**Pros:** Fully decoupled, extensible
**Cons:** Hard to debug, implicit dependencies

---

### Alternative 5: No Lookup - Flat Namespace via Solver Fields

Models are direct solver attributes. No lookup mechanism needed.

```python
class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, solver):  # Receive solver directly
        # Direct attribute access - IDE autocomplete works
        solver.velocity.use_buoyancy.configure(True, source=self.name)
        solver.velocity.g.configure(self.gravity, source=self.name)


class IcoFoamSolver(BaseModel):
    velocity: VelocityModel = VelocityModel()
    pressure: PressureModel = PressureModel()
    buoyancy: BuoyancyModel = BuoyancyModel()
```

**Pros:** Simple, type-safe, IDE support
**Cons:** Tight coupling to solver structure

---

### Comparison

| Approach | Coupling | Type Safety | Testability | Complexity |
|----------|----------|-------------|-------------|------------|
| ModelRegistry | Loose | Medium | High | Low |
| Solver Mediator | Tight | High | Medium | Medium |
| Dependency Injection | Explicit | High | High | Medium |
| Capability Queries | Loose | High | High | Medium |
| Event Bus | Very Loose | Low | Medium | High |
| Direct Solver Fields | Tight | High | Medium | Low |

---

### Recommendation

**Alternative 5 (Direct Solver Fields)** for simple cases:

```python
@Model.resolve_dependencies
def resolve(self, solver):
    solver.velocity.use_buoyancy.configure(True, source=self.name)
```

**ModelRegistry** when you need dynamic model lookup or plugins.

The key insight: Pass **solver** instead of **registry** to resolve methods. Models can access siblings directly via `solver.model_name`.
