# Initialization Framework Design Proposals

This document presents different design approaches for the 3-stage initialization framework, focusing on **usage examples** to help evaluate which API feels most natural and maintainable.

## Core Design Principles

The initialization framework follows three distinct stages with clear responsibilities:

1. **LOAD Stage**: Initialize Pydantic models from configuration files (pure data loading)
2. **RESOLVE_DEPENDENCIES Stage**: Modify model fields based on other models (inter-model configuration)
3. **BUILD Stage**: Return **lazy functions** that produce runtime objects, enabling DAG-based dependency resolution

The key insight is that BUILD should not execute initialization immediately, but return **deferred computations** that can be topologically sorted and executed in dependency order.

---

## Current Design Issues

```python
# PROBLEM: Current design executes build immediately
@Model.build
def build_fields(self, mesh, builder: ContextBuilder):
    # This runs immediately - no dependency resolution!
    builder.add_field("nu", create_scalar_field(mesh, self.nu))
```

**Issues:**
- LOAD does more than just initialize the Pydantic model
- BUILD executes immediately, can't handle complex dependencies
- No way to express "field X depends on field Y being created first"
- Order of model registration determines execution order (fragile)

---

## Proposed Design: Lazy BUILD with DAG Resolution

### Stage Responsibilities

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| **LOAD** | Config files | Pydantic model instance | Pure data loading into model fields |
| **RESOLVE** | ModelRegistry | Modified model fields | Inter-model configuration |
| **BUILD** | Mesh, Context | `List[LazyInit]` | Deferred initialization functions |

### Core Types

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class LazyInit:
    """A deferred initialization that declares its dependencies."""
    name: str                          # Unique identifier (e.g., "fields.U")
    depends_on: list[str]              # Dependencies (e.g., ["fields.p", "mesh"])
    initializer: Callable[[], Any]     # Lazy function to execute
    
    def execute(self) -> Any:
        """Execute the deferred initialization."""
        return self.initializer()
```

---

## Proposal 1: Explicit LazyInit Returns

Each BUILD method returns a list of `LazyInit` objects.

```python
from pydantic import BaseModel
from foamadapter.framework import Model, LazyInit

class TransportModel(BaseModel):
    """Transport properties - loaded from transportProperties file."""
    name: str = "transport"
    nu: float = None  # Populated during LOAD
    
    @Model.load
    def load(self):
        """LOAD: Initialize model from config files."""
        props = read_transport_properties()
        self.nu = props["nu"]
    
    # No resolve_dependencies needed - no inter-model config
    
    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        """BUILD: Return lazy initializers for runtime objects."""
        return [
            LazyInit(
                name="fields.nu",
                depends_on=["mesh"],
                initializer=lambda: create_scalar_field(mesh, self.nu)
            )
        ]


class PressureModel(BaseModel):
    """Pressure equation - depends on velocity field."""
    name: str = "pressure"
    p_ref: float = None
    
    @Model.load
    def load(self):
        self.p_ref = read_fv_solution()["pRefValue"]
    
    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            LazyInit(
                name="fields.p",
                depends_on=["mesh"],
                initializer=lambda: create_scalar_field(mesh, self.p_ref)
            ),
            LazyInit(
                name="operators.pressure_poisson",
                depends_on=["fields.p", "fields.U"],  # Depends on velocity!
                initializer=lambda: create_pressure_equation(mesh)
            )
        ]


class VelocityModel(BaseModel):
    """Velocity field and momentum equation."""
    name: str = "velocity"
    use_buoyancy: bool = False  # Can be modified by other models
    
    @Model.load
    def load(self):
        self.U0 = read_initial_conditions()["U"]
    
    @Model.resolve_dependencies
    def resolve(self, registry: ModelRegistry):
        """RESOLVE: Check if buoyancy model wants to modify us."""
        if registry.contains("buoyancy"):
            buoyancy = registry.get("buoyancy")
            self.use_buoyancy = buoyancy.enabled
    
    @Model.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            LazyInit(
                name="fields.U",
                depends_on=["mesh"],
                initializer=lambda: create_vector_field(mesh, self.U0)
            ),
            LazyInit(
                name="operators.momentum",
                depends_on=["fields.U", "fields.p", "fields.nu"],
                initializer=lambda: create_momentum_equation(
                    mesh, 
                    include_buoyancy=self.use_buoyancy
                )
            )
        ]


class IcoFoamSolver(BaseModel):
    """Incompressible Navier-Stokes solver."""
    transport: TransportModel = TransportModel()
    pressure: PressureModel = PressureModel()
    velocity: VelocityModel = VelocityModel()
    
    @Solver.load
    def load(self):
        self.dt = read_control_dict()["deltaT"]
        self.end_time = read_control_dict()["endTime"]
    
    @Solver.build
    def build(self, mesh) -> list[LazyInit]:
        return [
            LazyInit(
                name="mesh",
                depends_on=[],
                initializer=lambda: mesh  # Mesh is the root
            ),
            LazyInit(
                name="solver.time_loop",
                depends_on=["operators.momentum", "operators.pressure_poisson"],
                initializer=lambda: create_time_loop(self.dt, self.end_time)
            )
        ]


# Usage
solver = IcoFoamSolver()
initializer = SolverInitializer(solver)
ctx = initializer.initialize(mesh)

# What happens internally:
# 1. LOAD: All models load config into their fields
# 2. RESOLVE: Models modify each other's fields (e.g., buoyancy → velocity)
# 3. BUILD: Collect all LazyInit from all models
# 4. DAG RESOLUTION: Topologically sort by dependencies
# 5. EXECUTE: Run initializers in dependency order
```

**Pros:**
- Explicit dependency declaration
- DAG enables parallel initialization
- Clear separation: data (LOAD) → config (RESOLVE) → deferred init (BUILD)

**Cons:**
- Verbose LazyInit creation
- Must manually track dependency names

---

## Proposal 2: Decorator-based Lazy Registration

Use decorators to register lazy initializers with automatic dependency inference.

```python
from pydantic import BaseModel
from foamadapter.framework import Model, lazy, depends_on

class VelocityModel(BaseModel):
    name: str = "velocity"
    U0: Any = None
    
    @Model.load
    def load(self):
        self.U0 = read_initial_conditions()["U"]
    
    @Model.build
    @lazy("fields.U")
    @depends_on("mesh")
    def create_velocity_field(self, mesh):
        """This becomes a LazyInit automatically."""
        return create_vector_field(mesh, self.U0)
    
    @Model.build
    @lazy("operators.momentum")
    @depends_on("fields.U", "fields.p", "fields.nu")
    def create_momentum_equation(self, mesh):
        return create_momentum_equation(mesh)


class PressureModel(BaseModel):
    name: str = "pressure"
    
    @Model.load
    def load(self):
        self.p_ref = read_fv_solution()["pRefValue"]
    
    @Model.build
    @lazy("fields.p")
    @depends_on("mesh")
    def create_pressure_field(self, mesh):
        return create_scalar_field(mesh, self.p_ref)
    
    @Model.build
    @lazy("operators.pressure_poisson")
    @depends_on("fields.p", "fields.U")
    def create_pressure_equation(self, mesh):
        return create_pressure_equation(mesh)


# Usage - same as before
solver = IcoFoamSolver()
ctx = SolverInitializer(solver).initialize(mesh)
```

**Pros:**
- Decorators make dependencies visible at definition site
- Each method is a single LazyInit
- Natural Python style

**Cons:**
- Multiple decorators can be noisy
- Dependency names are strings (typos possible)

---

## Proposal 3: Context-Aware Lazy with Automatic Dependency Tracking

The framework tracks dependencies automatically based on what the initializer accesses.

```python
from pydantic import BaseModel
from foamadapter.framework import Model, InitContext

class VelocityModel(BaseModel):
    name: str = "velocity"
    
    @Model.load
    def load(self):
        self.U0 = read_initial_conditions()["U"]
    
    @Model.build
    def build(self, ctx: InitContext):
        """BUILD methods receive a context that tracks access."""
        
        # Register a lazy initializer - dependencies tracked automatically
        @ctx.lazy("fields.U")
        def create_U():
            mesh = ctx.get("mesh")  # Tracked as dependency!
            return create_vector_field(mesh, self.U0)
        
        @ctx.lazy("operators.momentum") 
        def create_momentum():
            U = ctx.get("fields.U")      # Dependency tracked
            p = ctx.get("fields.p")      # Dependency tracked
            nu = ctx.get("fields.nu")    # Dependency tracked
            return create_momentum_equation(U, p, nu)


class PressureModel(BaseModel):
    name: str = "pressure"
    
    @Model.load
    def load(self):
        self.p_ref = 0.0
    
    @Model.build
    def build(self, ctx: InitContext):
        @ctx.lazy("fields.p")
        def create_p():
            mesh = ctx.get("mesh")
            return create_scalar_field(mesh, self.p_ref)
        
        @ctx.lazy("operators.pressure_poisson")
        def create_pressure_eq():
            p = ctx.get("fields.p")
            U = ctx.get("fields.U")
            return create_pressure_equation(p, U)


# Usage
solver = IcoFoamSolver()
ctx = SolverInitializer(solver).initialize(mesh)

# Access triggers execution in dependency order
U = ctx.get("fields.U")  # Executes: mesh → fields.U
momentum = ctx.get("operators.momentum")  # Executes remaining deps
```

**Pros:**
- No manual dependency declaration
- Very clean syntax
- Impossible to have wrong dependencies

**Cons:**
- Magic dependency tracking (harder to debug)
- Must use `ctx.get()` instead of direct access

---

## Proposal 4: Hybrid - Explicit Dependencies with Helper Functions

Combine explicit LazyInit with helper functions for common patterns.

```python
from pydantic import BaseModel
from foamadapter.framework import Model, field, operator, lazy

class VelocityModel(BaseModel):
    name: str = "velocity"
    
    @Model.load
    def load(self):
        self.U0 = read_initial_conditions()["U"]
    
    @Model.build
    def build(self, mesh) -> list:
        return [
            # Helper for field creation
            field("U", 
                  depends_on=["mesh"],
                  create=lambda: create_vector_field(mesh, self.U0)),
            
            # Helper for operator creation
            operator("momentum",
                     depends_on=["fields.U", "fields.p", "fields.nu"],
                     create=lambda: create_momentum_equation(mesh)),
        ]


class TransportModel(BaseModel):
    name: str = "transport"
    
    @Model.load  
    def load(self):
        self.nu = read_transport_properties()["nu"]
    
    @Model.build
    def build(self, mesh) -> list:
        return [
            field("nu", create=lambda: create_scalar_field(mesh, self.nu))
        ]


class IcoFoamSolver(BaseModel):
    transport: TransportModel = TransportModel()
    velocity: VelocityModel = VelocityModel()
    pressure: PressureModel = PressureModel()
    
    @Solver.build
    def build(self, mesh) -> list:
        return [
            lazy("mesh", create=lambda: mesh),
            lazy("solver.loop", 
                 depends_on=["operators.momentum", "operators.pressure_poisson"],
                 create=lambda: TimeLoop(self.dt))
        ]
```

**Pros:**
- Clean helpers for common patterns
- `field()` auto-prefixes with "fields."
- `operator()` auto-prefixes with "operators."
- Still explicit about dependencies

**Cons:**
- Need to learn helper API

---

## DAG Resolution Implementation

Regardless of proposal, the DAG resolution works the same:

```python
class SolverInitializer:
    def initialize(self, mesh) -> Context:
        # Stage 1: LOAD - populate Pydantic models
        self._run_load()
        
        # Stage 2: RESOLVE - inter-model configuration
        self._run_resolve_dependencies()
        
        # Stage 3: BUILD - collect lazy initializers
        lazy_inits: list[LazyInit] = self._collect_lazy_inits(mesh)
        
        # Stage 4: DAG resolution
        execution_order = topological_sort(lazy_inits)
        
        # Stage 5: Execute in order
        context = {}
        for lazy_init in execution_order:
            result = lazy_init.execute()
            context[lazy_init.name] = result
        
        return Context(context)


def topological_sort(lazy_inits: list[LazyInit]) -> list[LazyInit]:
    """Sort lazy initializers by dependencies (Kahn's algorithm)."""
    # Build dependency graph
    graph = {li.name: li.depends_on for li in lazy_inits}
    
    # Find nodes with no dependencies
    ready = [li for li in lazy_inits if not li.depends_on]
    result = []
    
    while ready:
        current = ready.pop(0)
        result.append(current)
        
        # Find nodes that depended on current
        for li in lazy_inits:
            if current.name in li.depends_on:
                li.depends_on.remove(current.name)
                if not li.depends_on:
                    ready.append(li)
    
    if len(result) != len(lazy_inits):
        raise CyclicDependencyError("Circular dependency detected")
    
    return result
```

---

## Complete Example: icoFoam with Lazy BUILD

```python
from pydantic import BaseModel, Field
from foamadapter.framework import Model, Solver, field, operator, lazy

# ============================================================================
# Models
# ============================================================================

class TransportModel(BaseModel):
    """Laminar transport properties."""
    name: str = "transport"
    nu: float = Field(default=None, description="Kinematic viscosity")
    
    @Model.load
    def load(self):
        """LOAD: Read transportProperties dict."""
        props = read_dict("constant/transportProperties")
        self.nu = props["nu"]
    
    @Model.build
    def build(self, mesh):
        """BUILD: Create nu field (no dependencies except mesh)."""
        return [
            field("nu", create=lambda: volScalarField(mesh, self.nu))
        ]


class VelocityModel(BaseModel):
    """Velocity field and momentum equation."""
    name: str = "velocity"
    relax: float = Field(default=0.7, description="Under-relaxation factor")
    
    @Model.load
    def load(self):
        """LOAD: Read fvSolution for relaxation."""
        solution = read_dict("system/fvSolution")
        self.relax = solution.get("relaxationFactors", {}).get("U", 0.7)
    
    @Model.build
    def build(self, mesh):
        return [
            field("U", 
                  create=lambda: volVectorField(mesh, "U")),
            
            field("phi",
                  depends_on=["fields.U"],
                  create=lambda: createPhi(mesh)),
            
            operator("momentum",
                     depends_on=["fields.U", "fields.phi", "fields.nu", "fields.p"],
                     create=lambda: MomentumEquation(mesh, self.relax))
        ]


class PressureModel(BaseModel):
    """Pressure field and Poisson equation."""
    name: str = "pressure"
    n_correctors: int = Field(default=2, description="PISO corrector loops")
    n_non_ortho: int = Field(default=0, description="Non-orthogonal correctors")
    
    @Model.load
    def load(self):
        piso = read_dict("system/fvSolution")["PISO"]
        self.n_correctors = piso.get("nCorrectors", 2)
        self.n_non_ortho = piso.get("nNonOrthogonalCorrectors", 0)
    
    @Model.build
    def build(self, mesh):
        return [
            field("p",
                  create=lambda: volScalarField(mesh, "p")),
            
            operator("pressure_poisson",
                     depends_on=["fields.p", "fields.U", "fields.phi"],
                     create=lambda: PressureEquation(
                         mesh, self.n_correctors, self.n_non_ortho
                     ))
        ]


# ============================================================================
# Solver
# ============================================================================

class IcoFoamSolver(BaseModel):
    """Transient solver for incompressible, laminar flow."""
    
    # Sub-models
    transport: TransportModel = TransportModel()
    velocity: VelocityModel = VelocityModel()
    pressure: PressureModel = PressureModel()
    
    # Solver config (populated in LOAD)
    dt: float = Field(default=None)
    end_time: float = Field(default=None)
    write_interval: float = Field(default=None)
    
    @Solver.load
    def load(self):
        """LOAD: Read controlDict."""
        ctrl = read_dict("system/controlDict")
        self.dt = ctrl["deltaT"]
        self.end_time = ctrl["endTime"]
        self.write_interval = ctrl.get("writeInterval", self.dt)
    
    @Solver.build
    def build(self, mesh):
        return [
            lazy("mesh", create=lambda: mesh),
            
            lazy("runtime",
                 create=lambda: Time(self.dt, self.end_time)),
            
            lazy("piso_loop",
                 depends_on=[
                     "operators.momentum",
                     "operators.pressure_poisson",
                     "runtime"
                 ],
                 create=lambda: PISOLoop())
        ]
    
    def run(self, ctx: Context):
        """Main solve loop using initialized context."""
        piso = ctx.get("piso_loop")
        runtime = ctx.get("runtime")
        
        while runtime.loop():
            piso.solve()
            runtime.write()


# ============================================================================
# Usage
# ============================================================================

def main():
    # 1. Create mesh (OpenFOAM)
    mesh = create_mesh()
    
    # 2. Create and initialize solver
    solver = IcoFoamSolver()
    ctx = SolverInitializer(solver).initialize(mesh)
    
    # 3. What happened:
    #    LOAD:    All models read their config files
    #    RESOLVE: (none in this example)
    #    BUILD:   Collected 9 LazyInit objects
    #    DAG:     Sorted by dependencies
    #    EXECUTE: Created in order:
    #             mesh → runtime
    #             fields.nu → fields.U → fields.phi → fields.p
    #             operators.momentum → operators.pressure_poisson
    #             piso_loop
    
    # 4. Run simulation
    solver.run(ctx)
```

---

## Recommendation

**Proposal 4 (Hybrid with Helpers)** offers the best balance:

1. **LOAD** stays pure: just populate Pydantic model fields from files
2. **RESOLVE** modifies fields based on inter-model dependencies  
3. **BUILD** returns lazy initializers with explicit dependencies
4. Helper functions (`field()`, `operator()`, `lazy()`) reduce boilerplate
5. DAG resolution ensures correct initialization order

This design enables:
- Parallel initialization of independent fields
- Clear error messages for missing/circular dependencies
- Easy testing (can mock dependencies)
- Natural extension to async initialization

---

## Comparison Matrix

| Aspect | Proposal 1 (Explicit) | Proposal 2 (Decorators) | Proposal 3 (Auto-track) | Proposal 4 (Helpers) |
|--------|----------------------|-------------------------|------------------------|---------------------|
| **Verbosity** | High | Medium | Low | Low |
| **Explicitness** | Very High | High | Low | High |
| **Type Safety** | High | High | Medium | High |
| **Debuggability** | Easy | Easy | Hard | Easy |
| **Learning Curve** | Medium | Low | Medium | Low |
| **Flexibility** | High | High | High | High |

---

## Next Steps

1. Implement `LazyInit` dataclass
2. Add `field()`, `operator()`, `lazy()` helper functions
3. Implement topological sort in `SolverInitializer`
4. Update existing tests to use lazy BUILD pattern
5. Add cycle detection with helpful error messages
