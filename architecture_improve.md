# Architecture Improvement Proposals

## Current Architecture Overview

The current architecture implements a framework for CFD solvers with the following key components:

### Core Components
- **Solvers** (`@Solver` decorator): Define main simulation loop and operations
- **Models** (`@Model` decorator): Extend solver functionality with additional operations
- **Operations**: Decorated methods with dependency tracking and automatic context injection
- **Context**: Container for fields, models, mesh, and runtime
- **Algorithms**: Encapsulate pressure-velocity coupling strategies (PIMPLE, PISO, SIMPLE)

### Current Strengths
1. ✅ **Clean separation**: Algorithm logic separated from solver infrastructure
2. ✅ **Declarative operations**: `@Model.operation` and `@Solver.operation` with auto-injection
3. ✅ **Protocol-based design**: `PressureVelocityAlgorithm` enables polymorphism
4. ✅ **Context management**: Unified access to fields and models
5. ✅ **Automatic dependency resolution**: Operations track dependencies

---

## Proposed Improvements

### 1. **Eliminate Redundant Accessor Methods**

**Current Issue:**
```python
class PimpleAlgorithm:
    @Model.operation(operation_number=1)
    def momentum_operation(self, ...):
        # Implementation
    
    def momentum(self) -> Operation:
        """Return the momentum operation."""
        if self._momentum_op is None:
            ops = self.operations()
            for op in ops:
                if op.operation_name == "momentum_operation":
                    self._momentum_op = op
                    break
        return self._momentum_op
```

**Problem:** 
- Duplication: `momentum()` just searches for `momentum_operation`
- Caching logic scattered across multiple methods
- Error-prone: Operation name matching by string

**Proposed Solution:**
```python
class PimpleAlgorithm:
    @Model.operation(operation_number=1, name="momentum")
    def momentum(self, U, phi, p, turbulence, pimple_control: ModelAnnotation):
        """PIMPLE momentum: Assemble and solve momentum equation."""
        # Implementation directly in the method
        ...
    
    @Model.operation(operation_number=2, name="continuity")
    def continuity(self, U, p, phi, UEqn, pimple_control: ModelAnnotation):
        """PIMPLE continuity: Pressure-velocity coupling."""
        # Implementation directly in the method
        ...
```

**Benefits:**
- Single source of truth for each operation
- Natural access: `algorithm.momentum()` returns the Operation directly
- No manual caching or name matching needed
- Decorator can handle Operation wrapping transparently

**Implementation Note:** Would require updating the `@Model.operation` decorator to support storing Operations that can be retrieved by name.

---

### 2. **Improve Turbulence Model Integration**

**Current Issue:**
The solver currently handles turbulence correction manually:
```python
# In main_loop
time_loop.step(algorithm.momentum())
time_loop.step(algorithm.continuity())
# Turbulence correction is implicit somewhere
```

**Problem:**
- Turbulence correction responsibility is unclear
- Not visible in the operation graph
- Hard to customize or extend turbulence models

**Proposed Solution A - Turbulence as a Model:**
```python
@Model
class TurbulenceModel:
    @Model.operation(operation_number=1, depends_on=["continuity"])
    def correct_turbulence(self, laminarTransport, turbulence, pimple_control: ModelAnnotation):
        """Correct turbulence model."""
        if pimple_control.turbCorr():
            laminarTransport.correct()
            turbulence.correct()
        return FieldUpdates({
            "laminarTransport": laminarTransport,
            "turbulence": turbulence
        })
```

**Proposed Solution B - Algorithm-Owned Turbulence:**
```python
@Model
class PimpleAlgorithm:
    @Model.operation(operation_number=3, depends_on=["continuity"])
    def turbulence_correction(self, laminarTransport, turbulence, pimple_control: ModelAnnotation):
        """PIMPLE turbulence correction."""
        if pimple_control.turbCorr():
            laminarTransport.correct()
            turbulence.correct()
        return FieldUpdates({"laminarTransport": laminarTransport, "turbulence": turbulence})
```

**Benefits:**
- Explicit operation in the DAG
- Easier to extend (e.g., different turbulence correction strategies)
- Better separation of concerns
- Visible in operation visualization

**Recommendation:** Solution B - keeps turbulence correction with the algorithm since timing is algorithm-specific.

---

### 3. **Remove Unused Condition Classes**

**Current Issue:**
```python
class PimpleLoopCondition:
    """Condition for PIMPLE iterations."""
    def __init__(self) -> None:
        self.pimple = None
    def __call__(self, ctx: Any) -> bool:
        if self.pimple is None:
            self.pimple = ctx.models.get("pimple_control")
            if self.pimple is None:
                raise ValueError("pimple_control not found in context")
        return self.pimple.loop()

# Similar for PimpleCorrectorCondition, NonOrthogonalCondition
```

**Problem:**
- These classes are defined but never used
- The while loops in `continuity_operation` directly call `pimple_control.loop()`, `correct()`, etc.
- Dead code cluttering the module

**Proposed Solution:**
Remove these classes entirely or convert to utility functions if they'll be used for iterative operations.

**If keeping for future use:**
```python
def create_pimple_loop_condition(ctx: Context) -> Callable[[], bool]:
    """Factory for PIMPLE loop condition."""
    pimple_control = ctx.models.get("pimple_control")
    if pimple_control is None:
        raise ValueError("pimple_control not found in context")
    return pimple_control.loop
```

---

### 4. **Enhance Protocol with Type Hints**

**Current Issue:**
```python
def create_control(self, mesh: Any) -> Any:
    """Create algorithm control object."""
    ...
```

**Problem:**
- `Any` type loses type information
- No IDE support or type checking
- Unclear what control object types are expected

**Proposed Solution:**
```python
from typing import Protocol, TypeVar
from pybFoam import pimpleControl, pisoControl, simpleControl

ControlType = TypeVar('ControlType', pimpleControl, pisoControl, simpleControl)

class PressureVelocityAlgorithm(Protocol):
    def create_control(self, mesh) -> ControlType:
        """Create algorithm control object (pimpleControl, pisoControl, simpleControl)."""
        ...
```

**Benefits:**
- Better type checking
- IDE autocomplete support
- Self-documenting code

---

### 5. **Simplify Algorithm Operations Discovery**

**Current Issue:**
```python
def operations(self) -> OperationCollection:
    """Return algorithm-specific operations."""
    if self._ops is None:
        funcs = decorated_member_functions(self)
        self._ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            self._ops.add(op)
    return self._ops
```

**Problem:**
- Boilerplate repeated in every Model/Algorithm
- Manual caching required
- Easy to forget or implement incorrectly

**Proposed Solution:**
Add default implementation in `@Model` decorator:
```python
def Model(cls: type) -> type:
    """A class decorator to mark a class as a Model in the framework."""
    
    # Add default operations() implementation if not provided
    if not hasattr(cls, 'operations'):
        def operations(self) -> OperationCollection:
            if not hasattr(self, '_ops_cache'):
                funcs = decorated_member_functions(self)
                self._ops_cache = OperationCollection()
                for func in funcs:
                    op = Operation.create_SeqOp(func)
                    self._ops_cache.add(op)
            return self._ops_cache
        
        cls.operations = operations
    
    return cls
```

**Benefits:**
- DRY principle - don't repeat yourself
- Consistent behavior across all Models
- Users only override if custom behavior needed

---

### 6. **Add Algorithm Factory Pattern**

**Current Issue:**
```python
def _create_algorithm(self) -> PimpleAlgorithm:
    """Factory method to create algorithm instance."""
    if self.algorithm == "PIMPLE":
        return PimpleAlgorithm(pRefCell=self.pRefCell, pRefValue=self.pRefValue)
    else:
        raise ValueError(f"Algorithm '{self.algorithm}' not yet implemented.")
```

**Problem:**
- Hard-coded algorithm selection
- Not extensible without modifying solver code
- Violates Open-Closed Principle

**Proposed Solution A - Registry Pattern:**
```python
# In pressure_velocity.py
ALGORITHM_REGISTRY: dict[str, type[PressureVelocityAlgorithm]] = {}

def register_algorithm(name: str):
    """Decorator to register pressure-velocity algorithms."""
    def decorator(cls):
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator

@register_algorithm("PIMPLE")
@Model
class PimpleAlgorithm:
    ...

@register_algorithm("PISO")
@Model
class PisoAlgorithm:
    ...

# In solver
def _create_algorithm(self) -> PressureVelocityAlgorithm:
    """Factory method to create algorithm instance."""
    algorithm_cls = ALGORITHM_REGISTRY.get(self.algorithm)
    if algorithm_cls is None:
        raise ValueError(f"Unknown algorithm: {self.algorithm}")
    return algorithm_cls(pRefCell=self.pRefCell, pRefValue=self.pRefValue)
```

**Proposed Solution B - Plugin System:**
Use existing plugin system infrastructure:
```python
from foamadapter.core.plugin_system import ExtensibleConfig, register_plugin

class AlgorithmConfig(ExtensibleConfig):
    algorithm_type: str

register_plugin(AlgorithmConfig, "PIMPLE", PimpleAlgorithm)
register_plugin(AlgorithmConfig, "PISO", PisoAlgorithm)
```

**Benefits:**
- Easy to add new algorithms
- No solver modification needed
- Third-party algorithm extensions possible

---

### 7. **Improve Field Update Clarity**

**Current Issue:**
```python
return FieldUpdates({"U": U, "p": p, "phi": phi})
```

**Problem:**
- Not clear which fields were actually modified
- Returns fields that might not have changed
- Inefficient: triggers update notifications for unchanged fields

**Proposed Solution:**
```python
# Only return fields that were actually modified
return FieldUpdates({"U": U})  # p and phi are modified in-place via .assign()

# Or add a FieldModification tracking system
class FieldUpdates:
    def __init__(self, modified: dict[str, Any] = None, touched: set[str] = None):
        self.modified = modified or {}  # Fields with new objects
        self.touched = touched or set()  # Fields modified in-place
```

**Benefits:**
- Clear communication of what changed
- Potential performance optimization
- Better for debugging and logging

---

### 8. **Separate Algorithm Configuration from Solver**

**Current Issue:**
```python
class IncompressibleFluid(BaseModel):
    algorithm: Literal["SIMPLE", "PISO", "PIMPLE"] = "PIMPLE"
    pRefCell: int | None = None
    pRefValue: float | None = None
```

**Problem:**
- Algorithm parameters mixed with solver parameters
- Not extensible for algorithms with different configuration needs
- Tight coupling

**Proposed Solution:**
```python
class AlgorithmConfig(BaseModel):
    type: Literal["SIMPLE", "PISO", "PIMPLE"]
    pRefCell: int | None = None
    pRefValue: float | None = None
    # Algorithm-specific params can be added by subclasses

class PimpleConfig(AlgorithmConfig):
    type: Literal["PIMPLE"] = "PIMPLE"
    nOuterCorrectors: int = 1
    nCorrectors: int = 2
    nNonOrthogonalCorrectors: int = 0

class IncompressibleFluid(BaseModel):
    algorithm: AlgorithmConfig = Field(default_factory=lambda: PimpleConfig())
```

**Benefits:**
- Better encapsulation
- Each algorithm can define its own configuration
- Type-safe configuration access

---

### 9. **Add Operation Metadata for Visualization**

**Current Enhancement:**
Operations already support DAG visualization, but could be enhanced:

**Proposed Addition:**
```python
@Model.operation(
    operation_number=1,
    name="momentum",
    description="Assemble and solve momentum equation",
    inputs=["U", "phi", "p", "turbulence"],
    outputs=["UEqn"],
    computational_cost="high",  # For scheduling hints
    parallel=False  # Indicates if operation can be parallelized
)
def momentum(self, ...):
    ...
```

**Benefits:**
- Richer operation graphs
- Better documentation
- Potential for automatic optimization
- Resource allocation hints

---

### 10. **Consider Async/Await for I/O Operations**

**Current Issue:**
```python
def write_output(self, ctx: Context) -> None:
    """Write fields to disk."""
    runTime = ctx.runTime
    runTime.write(True)  # Blocking I/O
```

**Proposed Solution:**
```python
@Solver.operation(operation_number=13, depends_on=["turbulence_correction"])
async def write_output(self, ctx: Context) -> None:
    """Write fields to disk asynchronously."""
    runTime = ctx.runTime
    await runTime.write_async(True)
```

**Benefits:**
- Non-blocking I/O operations
- Better performance for large datasets
- Can overlap computation with I/O

**Note:** Would require significant framework changes and async support in pybFoam.

---

## Implementation Priority

### High Priority (Immediate Impact)
1. **Eliminate redundant accessor methods** - Reduces code duplication
2. **Add turbulence correction as explicit operation** - Better architecture
3. **Remove unused condition classes** - Code cleanup
4. **Default operations() implementation in @Model** - Reduces boilerplate

### Medium Priority (Architecture Improvements)
5. **Algorithm factory/registry pattern** - Better extensibility
6. **Separate algorithm configuration** - Better encapsulation
7. **Enhance protocol type hints** - Better type safety

### Low Priority (Nice to Have)
8. **Improve field update clarity** - Minor optimization
9. **Add operation metadata** - Enhanced documentation
10. **Async I/O operations** - Requires major changes

---

## Migration Strategy

For implementing these changes:

1. **Backward Compatibility**: Add new features alongside old ones
2. **Deprecation Warnings**: Mark old patterns as deprecated
3. **Documentation**: Update examples and guides
4. **Testing**: Ensure all tests pass with both old and new patterns
5. **Gradual Migration**: Allow transition period for users

---

## Example: Complete Improved Algorithm

```python
@register_algorithm("PIMPLE")
@Model
class PimpleAlgorithm:
    """PIMPLE algorithm for pressure-velocity coupling."""
    
    def __init__(self, config: PimpleConfig):
        self.config = config
    
    def name(self) -> str:
        return "PIMPLE"
    
    def create_control(self, mesh) -> pimpleControl:
        return pyf.pimpleControl(mesh)
    
    @Model.operation(
        operation_number=1,
        name="momentum",
        description="Assemble and solve momentum equation",
        inputs=["U", "phi", "p", "turbulence"],
        outputs=["UEqn"]
    )
    def momentum(
        self, 
        U: volVectorField, 
        phi: surfaceScalarField, 
        p: volScalarField,
        turbulence: incompressibleTurbulenceModel,
        pimple_control: ModelAnnotation[pimpleControl]
    ) -> FieldUpdates:
        """PIMPLE momentum: Assemble and solve momentum equation."""
        UEqn = fvVectorMatrix(fvm.ddt(U) + fvm.div(phi, U) + turbulence.divDevReff(U))
        UEqn.relax()
        
        if pimple_control.momentumPredictor():
            pyf.solve(UEqn + fvc.grad(p))
        
        return FieldUpdates({"UEqn": UEqn})
    
    @Model.operation(
        operation_number=2,
        name="continuity",
        description="Pressure-velocity coupling with nested loops",
        inputs=["U", "p", "phi", "UEqn"],
        outputs=["U", "p", "phi"]
    )
    def continuity(
        self,
        U: volVectorField,
        p: volScalarField,
        phi: surfaceScalarField,
        UEqn: fvVectorMatrix,
        pimple_control: ModelAnnotation[pimpleControl]
    ) -> FieldUpdates:
        """PIMPLE continuity: Pressure-velocity coupling."""
        while pimple_control.loop():
            while pimple_control.correct():
                # ... pressure-velocity coupling logic ...
                pass
        
        return FieldUpdates({"U": U, "p": p, "phi": phi})
    
    @Model.operation(
        operation_number=3,
        name="turbulence_correction",
        depends_on=["continuity"],
        description="Correct turbulence model",
        inputs=["laminarTransport", "turbulence"],
        outputs=["laminarTransport", "turbulence"]
    )
    def turbulence_correction(
        self,
        laminarTransport,
        turbulence,
        pimple_control: ModelAnnotation[pimpleControl]
    ) -> FieldUpdates:
        """PIMPLE turbulence correction."""
        if pimple_control.turbCorr():
            laminarTransport.correct()
            turbulence.correct()
        return FieldUpdates({
            "laminarTransport": laminarTransport,
            "turbulence": turbulence
        })
```

---

## Conclusion

The current architecture is solid and well-designed. The proposed improvements focus on:
- **Reducing boilerplate** through better defaults
- **Improving extensibility** through registries and plugins
- **Enhancing clarity** through better type hints and explicit operations
- **Maintaining flexibility** through backward compatibility

These changes would make the codebase more maintainable, easier to extend, and more user-friendly while preserving the strong foundation already in place.
