# Plan: Supporting Multiple Pressure-Velocity Coupling Algorithms

## Overview

This document outlines a plan to extend the NeoFOAM framework to support multiple pressure-velocity coupling algorithms (SIMPLE, PISO, PIMPLE) in a modular and extensible way.

## Implementation Steps

### Step 1: Create Protocol and PIMPLE Algorithm
1. Define `PressureVelocityAlgorithm` Protocol
2. Extract current PIMPLE loop into `PimpleAlgorithm` class
3. Update `IncompressibleFluid.main_loop()` to use algorithm
4. Verify existing test passes: `test_incompressible_fluid_pitzDaily.py`

### Step 2: Add SIMPLE Algorithm  
1. Implement `SimpleAlgorithm` class
2. Create new test: `test_incompressible_fluid_pitzDaily_steady.py`
   - Use `pitzDaily_steady` case (attached)
   - Test with `algorithm="SIMPLE"`
   - Verify steady-state convergence
3. Test should work exactly like current test but with SIMPLE algorithm

### Step 3: Add PISO Algorithm (optional)
1. Implement `PisoAlgorithm` class
2. Create test with transient case
3. Verify against OpenFOAM PISO results

## Proposed API Usage

The main improvement is how the `main_loop()` method changes to support multiple algorithms:

### Current Implementation (PIMPLE hardcoded)
```python
def main_loop(self, ctx: Context) -> None:
    """Main simulation loop with PIMPLE algorithm structure hardcoded."""
    ops = self.operations()
    
    main_loop = StepBuilder()
    time_loop_op = Operation(...)
    
    with main_loop.loop(time_loop_op) as time_loop:
        time_loop.step(ops["print_time"])
        
        # PIMPLE structure hardcoded here - 60+ lines of nested loops
        pimple_loop_op = Operation(...)
        with time_loop.loop(pimple_loop_op) as pimple_loop:
            pimple_loop.step(ops["momentum_predictor"])
            # ... more nesting ...
        
        time_loop.step(ops["write_output"])
    
    main_loop.operations.run(ctx)
```

### New Implementation (Algorithm selectable)
```python
def main_loop(self, ctx: Context) -> None:
    """Main simulation loop - algorithm agnostic!"""
    ops = self.operations()
    algorithm = self._create_algorithm()  # SIMPLE, PISO, or PIMPLE
    
    main_loop = StepBuilder()
    time_loop_op = Operation(...)
    
    with main_loop.loop(time_loop_op) as time_loop:
        time_loop.step(ops["print_time"])
        
        # Algorithm returns its momentum and continuity operations
        time_loop.step(algorithm.momentum)      # Just an operation
        time_loop.step(algorithm.continuity)    # Just an operation
        
        time_loop.step(ops["write_output"])
    
    main_loop.operations.run(ctx)
```

**Key Benefits**:
- `main_loop()` reduced from ~80 lines to ~15 lines
- Algorithm structure explicit in algorithm class, not hidden in solver
- Easy to switch: just change `algorithm = "PIMPLE"` to `algorithm = "PISO"`
- Same operations reused by all algorithms
- Clean API: `algorithm.momentum` and `algorithm.continuity` are just operations

## Current State

### Existing Implementation
- **IncompressibleFluid solver**: Implements PIMPLE algorithm with hardcoded loop structure
- **IcoFoam solver**: Simple PISO implementation without framework integration
- **Loop structure**: 4 nested levels (time → PIMPLE → corrector → non-orthogonal)
- **Operations**: 13 sequential operations with explicit dependencies
- **Condition classes**: CFLCondition, PimpleLoopCondition, PimpleCorrectorCondition, NonOrthogonalCondition

### Limitations
1. Algorithm is hardcoded in `main_loop()` method
2. No easy way to switch between SIMPLE/PISO/PIMPLE
3. Loop conditions are algorithm-specific classes
4. Operation sequence is fixed for one algorithm type

## Design Goals

### Primary Goals
1. **Modularity**: Separate algorithm logic from solver operations
2. **Reusability**: Share common operations across algorithms
3. **Configurability**: Select algorithm via configuration/runtime parameter
4. **Extensibility**: Easy to add new algorithms (e.g., SIMPLEC, PIMPLEC)
5. **Framework Compliance**: Maintain compatibility with existing NeoFOAM framework

### Secondary Goals
1. Minimal code duplication across algorithms
2. Clear separation of concerns
3. Maintain performance characteristics
4. Preserve existing test compatibility

## Proposed Architecture

### 1. Algorithm Protocol Interface

#### 1.1 Algorithm Protocol

```python
from typing import Protocol, runtime_checkable
from foamadapter.framework.operations import Operation

@runtime_checkable
class PressureVelocityAlgorithm(Protocol):
    """
    Protocol defining the interface for pressure-velocity coupling algorithms.
    
    Algorithms expose momentum and continuity as operation properties that can
    be directly used in the solver's time loop.
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
    
    @property
    def momentum(self) -> Operation:
        """
        Momentum prediction operation.
        
        Returns a single operation (possibly with nested loops) that handles
        the momentum prediction phase of the algorithm.
            
        Example:
            PIMPLE/PISO/SIMPLE: momentum_predictor + solve_momentum (sequential)
        """
        ...
    
    @property
    def continuity(self) -> Operation:
        """
        Continuity/pressure correction operation.
        
        This is the core of the algorithm, defining the pressure-velocity coupling
        strategy. Returns a single operation that may contain nested loops.
            
        Example:
            PISO: corrector_loop(compute_HbyA -> ... -> correct_velocity)
            PIMPLE: pimple_loop(corrector_loop(...) + turbulence)
            SIMPLE: compute_HbyA -> ... -> correct_velocity (sequential)
        """
        ...
```


#### 1.2 Concrete Algorithm Implementations

Algorithms build their loop structure using the same operations, just arranged differently.
This stays very close to the current implementation - just extracted into reusable classes.

**PIMPLE Algorithm** (current implementation):
```python
class PimpleAlgorithm:
    """PIMPLE algorithm - matches current IncompressibleFluid implementation."""
    
    def __init__(self, operations: OperationCollection):
        """Initialize with solver operations."""
        self.ops = operations
    
    def name(self) -> str:
        return "PIMPLE"
    
    def create_control(self, mesh: Any) -> Any:
        return pyf.pimpleControl(mesh)
    
    @property
    def momentum(self) -> Operation:
        """
        PIMPLE momentum: Simple sequential execution.
        Returns: momentum_predictor -> solve_momentum
        """
        builder = StepBuilder()
        builder.step(self.ops["momentum_predictor"])
        builder.step(self.ops["solve_momentum"])
        return Operation.from_collection(builder.operations, "pimple_momentum")
    
    @property
    def continuity(self) -> Operation:
        """
        PIMPLE continuity with nested loops (exactly as current implementation).
        
        Structure:
        - PIMPLE outer loop
          - Corrector loop
            - Compute HbyA, phiHbyA, adjust phi
            - Non-orthogonal loop
              - Solve pressure, update flux
            - Correct velocity
          - Turbulence correction
        """
        builder = StepBuilder()
        
        # Outer PIMPLE loop
        pimple_loop_op = Operation(
            func=IterativeOp(PimpleLoopCondition()),
            operation_name="pimple_loop",
            operation_number=1,
        )
        
        with builder.loop(pimple_loop_op) as pimple_loop:
            # Corrector loop
            corrector_op = Operation(
                func=IterativeOp(PimpleCorrectorCondition()),
                operation_name="corrector_loop",
                operation_number=2,
            )
            
            with pimple_loop.loop(corrector_op) as corrector_loop:
                corrector_loop.step(self.ops["compute_HbyA"])
                corrector_loop.step(self.ops["compute_phiHbyA"])
                corrector_loop.step(self.ops["adjust_phi"])
                
                # Non-orthogonal loop
                non_orth_op = Operation(
                    func=IterativeOp(NonOrthogonalCondition()),
                    operation_name="non_orth_loop",
                    operation_number=3,
                )
                
                with corrector_loop.loop(non_orth_op) as non_orth:
                    non_orth.step(self.ops["solve_pressure"])
                    non_orth.step(self.ops["update_flux"])
                
                corrector_loop.step(self.ops["correct_velocity"])
            
            pimple_loop.step(self.ops["turbulence_correction"])
        
        return Operation.from_collection(builder.operations, "pimple_continuity")
```

**PISO Algorithm**:
```python
class PisoAlgorithm:
    """PISO algorithm - simpler than PIMPLE, no outer loop."""
    
    def __init__(self, operations: OperationCollection):
        """Initialize with solver operations."""
        self.ops = operations
    
    def name(self) -> str:
        return "PISO"
    
    def create_control(self, mesh: Any) -> Any:
        return pyf.pisoControl(mesh)
    
    @property
    def momentum(self) -> Operation:
        """
        PISO momentum: Simple sequential execution.
        Returns: momentum_predictor -> solve_momentum
        """
        builder = StepBuilder()
        builder.step(self.ops["momentum_predictor"])
        builder.step(self.ops["solve_momentum"])
        return Operation.from_collection(builder.operations, "piso_momentum")
    
    @property
    def continuity(self) -> Operation:
        """
        PISO continuity: Corrector loop only (no outer PIMPLE loop).
        
        Structure:
        - Corrector loop
          - Compute HbyA, phiHbyA, adjust phi
          - Non-orthogonal loop
            - Solve pressure, update flux
          - Correct velocity
        """
        builder = StepBuilder()
        
        # Corrector loop (no outer PIMPLE loop)
        corrector_op = Operation(
            func=IterativeOp(PisoCorrectorCondition()),
            operation_name="piso_corrector_loop",
            operation_number=1,
        )
        
        with builder.loop(corrector_op) as corrector_loop:
            corrector_loop.step(self.ops["compute_HbyA"])
            corrector_loop.step(self.ops["compute_phiHbyA"])
            corrector_loop.step(self.ops["adjust_phi"])
            
            non_orth_op = Operation(
                func=IterativeOp(NonOrthogonalCondition()),
                operation_name="non_orth_loop",
                operation_number=2,
            )
            
            with corrector_loop.loop(non_orth_op) as non_orth:
                non_orth.step(self.ops["solve_pressure"])
                non_orth.step(self.ops["update_flux"])
            
            corrector_loop.step(self.ops["correct_velocity"])
        
        return Operation.from_collection(builder.operations, "piso_continuity")
```

**SIMPLE Algorithm**:
```python
class SimpleAlgorithm:
    """SIMPLE algorithm - single correction, mostly sequential."""
    
    def __init__(self, operations: OperationCollection):
        """Initialize with solver operations."""
        self.ops = operations
    
    def name(self) -> str:
        return "SIMPLE"
    
    def create_control(self, mesh: Any) -> Any:
        return pyf.simpleControl(mesh)
    
    @property
    def momentum(self) -> Operation:
        """
        SIMPLE momentum: Simple sequential execution.
        Returns: momentum_predictor -> solve_momentum
        """
        builder = StepBuilder()
        builder.step(self.ops["momentum_predictor"])
        builder.step(self.ops["solve_momentum"])
        return Operation.from_collection(builder.operations, "simple_momentum")
    
    @property
    def continuity(self) -> Operation:
        """
        SIMPLE continuity: Mostly sequential with optional non-orthogonal loop.
        
        Structure:
        - Compute HbyA, phiHbyA, adjust phi
        - Non-orthogonal loop (optional)
          - Solve pressure, update flux
        - Correct velocity
        - Turbulence correction
        """
        builder = StepBuilder()
        
        builder.step(self.ops["compute_HbyA"])
        builder.step(self.ops["compute_phiHbyA"])
        builder.step(self.ops["adjust_phi"])
        
        # Optional non-orthogonal corrections
        non_orth_op = Operation(
            func=IterativeOp(NonOrthogonalCondition()),
            operation_name="non_orth_loop",
            operation_number=1,
        )
        
        with builder.loop(non_orth_op) as non_orth:
            non_orth.step(self.ops["solve_pressure"])
            non_orth.step(self.ops["update_flux"])
        
        builder.step(self.ops["correct_velocity"])
        builder.step(self.ops["turbulence_correction"])
        
        return Operation.from_collection(builder.operations, "simple_continuity")
```
            
            corrector_loop.step(atomic_ops["correct_velocity"])
        
        return Operation.create_SeqOp_from_collection(
            builder.operations,
            "piso_continuity"
        )
```

**SIMPLE Algorithm**:
```python
class SimpleAlgorithm:
    """SIMPLE algorithm: steady-state with under-relaxation."""
    
    def name(self) -> str:
        return "SIMPLE"
    
    def create_control(self, mesh: Any, ctx: Context) -> Any:
        control = pyf.simpleControl(mesh)
        ctx.models["simple_control"] = control
        return control
    
    def momentum_operation(
        self,
        atomic_ops: OperationCollection,
        ctx: Context
    ) -> Operation:
        """
        SIMPLE momentum: predictor + under-relaxation + solve
        """
        momentum = OperationCollection()
        momentum.add(atomic_ops["momentum_predictor"])
        momentum.add(atomic_ops["under_relax_momentum"])  # SIMPLE-specific
        momentum.add(atomic_ops["solve_momentum"])
        
        return Operation.create_SeqOp_from_collection(momentum, "simple_momentum")
    
    def continuity_operation(
        self,
        atomic_ops: OperationCollection,
        ctx: Context
    ) -> Operation:
        """
        SIMPLE continuity: single pressure correction with under-relaxation
        
        Structure:
        - No loops (or single iteration)
        - Under-relaxation applied
        """
        builder = StepBuilder()
        
        # SIMPLE typically has optional non-orthogonal corrections
        non_orth_op = Operation(
            func=IterativeOp(NonOrthogonalCondition()),
            operation_name="non_orth_loop",
            operation_number=1,
        )
        
        builder.step(atomic_ops["compute_HbyA"])
        builder.step(atomic_ops["compute_phiHbyA"])
        builder.step(atomic_ops["adjust_phi"])
        
        with builder.loop(non_orth_op) as non_orth:
            non_orth.step(atomic_ops["solve_pressure"])
            non_orth.step(atomic_ops["update_flux"])
        
        builder.step(atomic_ops["under_relax_pressure"])  # SIMPLE-specific
        builder.step(atomic_ops["correct_velocity"])
        builder.step(atomic_ops["under_relax_velocity"])  # SIMPLE-specific
        
        return Operation.create_SeqOp_from_collection(
            builder.operations,
            "simple_continuity"
        )
```


### 2. Unified Condition System

#### 2.1 Generic Condition Classes

```python
class AlgorithmLoopCondition:
    """Generic outer loop condition for any algorithm."""
    
    def __init__(self, algorithm: PressureVelocityAlgorithm):
        self.algorithm = algorithm
        self.control = None
    
    def __call__(self, ctx: Context) -> bool:
        if self.control is None:
            self.control = ctx.models.get(f"{self.algorithm.name()}_control")
        return self.control.loop()

class CorrectorLoopCondition:
    """Generic corrector loop condition."""
    
    def __init__(self, algorithm: PressureVelocityAlgorithm):
        self.algorithm = algorithm
        self.control = None
    
    def __call__(self, ctx: Context) -> bool:
        if self.control is None:
            self.control = ctx.models.get(f"{self.algorithm.name()}_control")
        return self.control.correct()
```

### 3. Refactored Solver Structure

#### 3.1 Solver Operations (Same as Current Implementation)

The solver keeps the **exact same operations** as the current IncompressibleFluid,
just organized to be reusable by different algorithms:

**Current Operations** (operations 1-13, unchanged):
1. `create_fields`: Field initialization
2. `setup_models`: Model configuration  
3. `print_time`: Print current simulation time
4. `momentum_predictor`: Assemble momentum equation (UEqn)
5. `solve_momentum`: Solve momentum equation
6. `compute_HbyA`: Compute H/A from momentum equation
7. `compute_phiHbyA`: Compute flux from H/A
8. `adjust_phi`: Adjust flux for continuity and constrain pressure
9. `solve_pressure`: Solve pressure equation
10. `update_flux`: Update flux after pressure solution
11. `correct_velocity`: Correct velocity field from pressure gradient
12. `turbulence_correction`: Update turbulence model
13. `write_output`: I/O operations

**Key Point**: Operations stay exactly as they are - same code, same dependencies,
same behavior. Only the `main_loop()` method changes to delegate loop building
to the algorithm.

#### 3.2 Minimal Solver Changes

The IncompressibleFluid solver changes **only the main_loop() method** - everything
else stays the same:

```python
@Solver
class IncompressibleFluid(BaseModel):
    """
    Incompressible fluid solver supporting multiple pressure-velocity algorithms.
    
    Exactly the same as current implementation, just with configurable algorithm.
    """
    
    name: Literal["IncompressibleFluid"] = "IncompressibleFluid"
    argv: list[str] = []
    algorithm: Literal["SIMPLE", "PISO", "PIMPLE"] = "PIMPLE"  # NEW: select algorithm
    pRefCell: int | None = None
    pRefValue: float | None = None
    maxDeltaT: float = 1e5
    
    def _create_algorithm(self) -> PressureVelocityAlgorithm:
        """Factory method to create algorithm instance."""
        ops = self.operations()
        if self.algorithm == "SIMPLE":
            return SimpleAlgorithm(ops)
        elif self.algorithm == "PISO":
            return PisoAlgorithm(ops)
        elif self.algorithm == "PIMPLE":
            return PimpleAlgorithm(ops)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def create_context(self) -> Context:
        """Initialize simulation context - UNCHANGED."""
        # ... exact same as current implementation ...
    
    @Solver.operation(operation_number=1)
    def create_fields(self, ctx: Context) -> FieldUpdates:
        """Create and read fields - UNCHANGED."""
        # ... exact same as current implementation ...
    
    @Solver.operation(operation_number=2, depends_on=["create_fields"])
    def setup_models(self, ctx: Context) -> None:
        """Setup models - MODIFIED to store algorithm control."""
        # Move pimple/piso/simple control to models
        algorithm = self._create_algorithm()
        control = algorithm.create_control(ctx.mesh)
        ctx.models[f"{algorithm.name().lower()}_control"] = control
        # Note: condition classes updated to look up control by algorithm name
    
    # Operations 3-13: UNCHANGED (exact same code as current implementation)
    @Solver.operation(operation_number=3, depends_on=["setup_models"])
    def print_time(self, ctx: Context) -> None: ...
    
    @Solver.operation(operation_number=4, depends_on=["print_time"])
    def momentum_predictor(self, U, phi, turbulence) -> FieldUpdates: ...
    
    # ... all other operations stay exactly the same ...
    
    def operations(self, domain_name: str | None = None) -> OperationCollection:
        """Collect all operations - UNCHANGED."""
        _ = domain_name
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops
    
    def main_loop(self, ctx: Context) -> None:
        """
        Main simulation loop - ONLY CHANGED METHOD!
        
        Algorithm provides momentum and continuity as operation properties.
        """
        ops = self.operations()
        algorithm = self._create_algorithm()
        
        # Build time loop
        main_loop = StepBuilder()
        
        time_loop_op = Operation(
            func=IterativeOp(CFLCondition(self.maxDeltaT)),
            operation_name="time_loop",
            operation_number=1,
        )
        
        with main_loop.loop(time_loop_op) as time_loop:
            time_loop.step(ops["print_time"])
            time_loop.step(algorithm.momentum)      # Algorithm's momentum operation
            time_loop.step(algorithm.continuity)    # Algorithm's continuity operation
            time_loop.step(ops["write_output"])
        
        # Execute
        main_loop.operations.run(ctx)
    
    def run(self) -> None:
        """Run the complete simulation - UNCHANGED."""
        ctx = self.create_context()
        
        ops = self.operations()
        ops["create_fields"].run(ctx)
        ops["setup_models"].run(ctx)
        
        Info("Starting time loop")
        self.main_loop(ctx)
        Info("End")
```

**Changes Summary**:
1. ✅ Add `algorithm` parameter to select SIMPLE/PISO/PIMPLE
2. ✅ Modify `setup_models()` to use algorithm's `create_control()`
3. ✅ Replace hardcoded loop structure in `main_loop()` with `algorithm.build_algorithm_loop()`
4. ✅ Update condition classes to look up control by algorithm name
5. ✅ **All 13 operations stay exactly the same**

**Lines of code changed**: ~30 lines (main_loop + setup_models + factory)
**Lines of code unchanged**: ~350 lines (all operations + everything else)

### 4. Configuration and Control

#### 4.1 Algorithm Selection Methods

**Option A: Constructor Parameter** (Recommended for Python API)
```python
solver = IncompressibleFluid(
    argv=["incompressibleFluid"],
    algorithm="PIMPLE"
)
```

**Option B: ControlDict Configuration**
```cpp
// system/controlDict
application     incompressibleFluid;

algorithm       PIMPLE;  // or SIMPLE, PISO

PIMPLE
{
    nOuterCorrectors    2;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
}
```

**Option C: Command Line Argument**
```bash
incompressibleFluid -algorithm PIMPLE
```

#### 4.2 Algorithm-Specific Parameters

Each algorithm has its own control parameters stored in fvSolution:

```cpp
// system/fvSolution
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        p               1e-5;
        U               1e-5;
    }
}

PISO
{
    nCorrectors              2;
    nNonOrthogonalCorrectors 0;
}

PIMPLE
{
    nOuterCorrectors         2;
    nCorrectors              2;
    nNonOrthogonalCorrectors 1;
    outerCorrectorResidualControl
    {
        p               1e-4;
        U               1e-4;
    }
}
```

## Implementation Roadmap

### Step 1: Protocol & PIMPLE (Week 1)
1. Create `foamadapter/algorithms/pressure_velocity.py`
   - Define `PressureVelocityAlgorithm` Protocol
   - Implement `PimpleAlgorithm` class
   
2. Update `IncompressibleFluid` solver
   - Add `algorithm` parameter (default="PIMPLE")
   - Extract loop logic from `main_loop()` to `PimpleAlgorithm`
   - Update `setup_models()` to use algorithm's `create_control()`
   
3. Verify with existing test
   - Run `test_incompressible_fluid_pitzDaily.py`
   - Should pass without changes

### Step 2: SIMPLE Algorithm (Week 1-2)
1. Implement `SimpleAlgorithm` class in `pressure_velocity.py`
   - Sequential structure (no outer PIMPLE loop)
   - Uses `simpleControl` instead of `pimpleControl`
   
2. Create new test: `test_incompressible_fluid_pitzDaily_steady.py`
   ```python
   def test_incompressible_fluid_pitzDaily_steady():
       \"\"\"Test IncompressibleFluid with SIMPLE algorithm on steady case.\"\"\"
       from foamadapter.solver import IncompressibleFluid
       
       # Use pitzDaily_steady case
       source_case = repo_root / "tutorials" / "pitzDaily_steady"
       
       # Create solver with SIMPLE algorithm
       solver = IncompressibleFluid(
           argv=["incompressibleFluid"],
           algorithm="SIMPLE"
       )
       
       solver.run()
       
       # Verify convergence and output
       ...
   ```
   
3. Run both tests to verify:
   - PIMPLE on transient pitzDaily
   - SIMPLE on steady pitzDaily_steady

### Step 3: PISO Algorithm (Optional, Week 2)
1. Implement `PisoAlgorithm` class
2. Create test with transient cavity case
3. Verify all three algorithms work

## File Structure

```
src/foamadapter/
  algorithms/
    __init__.py
    pressure_velocity.py          # Protocol + PIMPLE/SIMPLE/PISO classes
  
  solver/
    incompressibleFluid.py        # Modified to use algorithm
    
test/solver/
  test_incompressible_fluid.py                    # Unit tests (unchanged)
  test_incompressible_fluid_pitzDaily.py          # PIMPLE integration test (unchanged)
  test_incompressible_fluid_pitzDaily_steady.py   # NEW: SIMPLE integration test
```

## Testing Strategy

### Unit Tests
- Test each algorithm class independently
- Verify loop structure construction
- Test condition evaluation logic
- Validate control object creation

### Integration Tests
- Run pitzDaily with all three algorithms
- Compare results for consistency
- Verify convergence characteristics
- Performance comparison

### Regression Tests
- Ensure existing PIMPLE tests still pass
- Validate backward compatibility
- Check output field accuracy

## Migration Path for Existing Code

### Backward Compatibility
```python
# Old code - still works
solver = IncompressibleFluid(argv=["solver"])
solver.run()  # Uses PIMPLE by default

# New code - explicit algorithm selection
solver = IncompressibleFluid(
    argv=["solver"],
    algorithm="PISO"
)
solver.run()
```

### Deprecation Strategy
1. Keep current implementation as default (PIMPLE)
2. Add warnings for future changes
3. Gradual migration over multiple releases
4. Clear documentation of changes

## Benefits of Proposed Design

### For Users
1. **Easy algorithm selection** without code changes
2. **Consistent interface** - same solver, different algorithms
3. **Better configuration options** via controlDict or constructor
4. **Clear documentation** per algorithm

### For Developers
1. **Dramatic code reduction** - IncompressibleFluid ~50% smaller
2. **No code duplication** - algorithms reuse atomic operations
3. **Explicit algorithm logic** - all in one place (algorithm class)
4. **Easy to add new algorithms** - just implement Protocol
5. **Better testability** - test atomic operations and algorithms separately

### For Framework
1. **Demonstrates Protocol-based extensibility** pattern
2. **Establishes operation composition** pattern for other solvers
3. **Improves overall architecture** - clear separation of concerns
4. **Better aligned with OpenFOAM structure** - control classes map directly

### Architecture Benefits
1. **Composition over inheritance** - algorithms compose operations
2. **Dependency injection** - algorithm receives operations, returns composed operations
3. **Single Responsibility** - solver provides operations, algorithm defines structure
4. **Open/Closed Principle** - open for extension (new algorithms), closed for modification (solver stays same)

## Potential Challenges

### Technical Challenges
1. **Operation Dependencies**: Some operations may need different dependencies per algorithm
2. **State Management**: Ensuring correct context state across different loop structures
3. **Performance**: Abstraction overhead vs. performance requirements
4. **Turbulence Models**: Integration with different algorithm characteristics

### Mitigation Strategies
1. Use dependency injection for algorithm-specific behavior
2. Carefully design context state updates
3. Profile and optimize hot paths
4. Document turbulence model compatibility per algorithm

## Future Extensions

### Additional Algorithms
- **SIMPLEC**: Enhanced SIMPLE with improved convergence
- **PIMPLEC**: PIMPLE with consistent formulation
- **Coupled**: Fully coupled pressure-velocity solver
- **Segregated**: Custom segregated approaches

### Advanced Features
- **Adaptive Algorithm Selection**: Switch algorithms based on convergence
- **Hybrid Approaches**: Combine algorithms in different regions
- **Custom Loop Structures**: User-defined algorithm variants
- **Multi-Physics Coupling**: Integrate with other physics solvers

## References

### OpenFOAM Documentation
- OpenFOAM User Guide: SIMPLE, PISO, PIMPLE algorithms
- OpenFOAM Programmer's Guide: pimpleControl, pisoControl, simpleControl
- Source code: `src/finiteVolume/cfdTools/general/solutionControl/`

### Academic Papers
- Patankar & Spalding (1972): SIMPLE algorithm
- Issa (1986): PISO algorithm
- Issa et al. (1991): PIMPLE algorithm

### NeoFOAM Framework
- Framework documentation
- Existing solver implementations
- Operation and context patterns

## Conclusion

This plan provides a structured approach to supporting multiple pressure-velocity coupling algorithms in the NeoFOAM framework. The modular design ensures:

1. **Maintainability**: Clear separation between algorithm logic and operations
2. **Extensibility**: Easy addition of new algorithms
3. **Usability**: Simple configuration and algorithm selection
4. **Performance**: Minimal overhead from abstraction layers

The phased implementation allows for incremental development and testing, ensuring stability and backward compatibility throughout the process.
