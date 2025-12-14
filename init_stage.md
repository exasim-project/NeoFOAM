# Solver Input Validation and Initialization Plan

## Overview

This document outlines the strategy for implementing input validation and staged initialization in NeoFOAM solvers. The plan builds on the existing Pydantic-based validation infrastructure and extends the solver framework to support robust, user-friendly initialization.

## Architecture Context

The initialization process follows a **3-stage model-centric approach**:

1. **Stage 1 - READ_FILES**: Read input files and initialize model objects
2. **Stage 2 - CONFIGURE**: Models configure themselves based on other models (dynamic operation selection)
3. **Stage 3 - SETUP**: Prepare models for execution before the main run loop

After initialization, the main simulation proceeds:
4. **Stage 4**: Executing operations (main loop)
5. **Stage 5**: Writing results

This plan focuses on **Stages 1-3**, defining how models validate inputs, configure themselves dynamically, and prepare for execution.

---

## The 3-Stage Initialization Model

### Stage 1: READ_FILES
**Purpose**: Read input files from disk and instantiate model objects

- Parse configuration files (controlDict, fvSchemes, fvSolution, etc.)
- Validate file existence and format
- Create Pydantic model instances from file data
- Initialize mesh, runtime, and basic data structures
- **No inter-model dependencies at this stage**

### Stage 2: CONFIGURE
**Purpose**: Models configure themselves based on other models

- Models can inspect other models to decide their behavior
- Dynamic operation selection (e.g., turbulence model returns different operations based on switches)
- Algorithm selection based on solver configuration
- **Triggered by other models** - configuration can cascade
- Build dependency graph based on configured operations

### Stage 3: SETUP
**Purpose**: Prepare models for execution before the main run loop

- Allocate runtime resources
- Initialize solver matrices and data structures
- Perform pre-computation (e.g., geometric calculations)
- Validate inter-model consistency
- **All models must be fully configured before setup**

---

## Current State Analysis

### What Already Exists

1. **Pydantic BaseModel**: All solvers inherit from `BaseModel` for configuration validation
   - Example: `IncompressibleFluid` with fields `name`, `argv`, `algorithm`, `pRefCell`, etc.
   - Automatic type checking and field validation

2. **Input Validation Infrastructure** (`foamadapter/io/input_validation.py`):
   - `ModelInputDefinition`: Associates a Pydantic model with a file path
   - `ModelInputCollection`: Registry of all input definitions
   - `ValidationErrors`: Structured error reporting (field, file, error_type, message)
   - `validate_case()`: Validates all required files exist and parse correctly

3. **Context Creation**: Solvers implement `create_context()` to build runtime state
   - Example: `IncompressibleFluid.create_context()` reads mesh, creates fields

4. **Operation Framework**: Decorated operations with dependency tracking
   - `@Solver.operation(operation_number=X, depends_on=[...])`
   - Automatic context injection via type hints

### What's Missing

1. **3-Stage Initialization Protocol**: No standard interface for read → configure → setup
2. **Dynamic Configuration**: Models cannot reconfigure based on other models
3. **Configuration Triggers**: No mechanism for models to trigger reconfiguration in others
4. **Setup Phase**: No explicit preparation phase before main loop
5. **Inter-Model Communication**: Models cannot query each other during initialization

---

## Proposed Solution

### Core Concept: Model Lifecycle Hooks

Each model (Solver, Algorithm, TurbulenceModel, etc.) implements 3 lifecycle methods:

```python
class ModelLifecycle(Protocol):
    """Protocol for model initialization lifecycle."""
    
    def read_files(self, case_dir: Path) -> None:
        """Stage 1: Read input files and initialize object state."""
        ...
    
    def configure(self, models: dict[str, Any]) -> None:
        """Stage 2: Configure based on other models. May modify operations."""
        ...
    
    def setup(self, ctx: Context) -> None:
        """Stage 3: Prepare for execution. Allocate resources, pre-compute."""
        ...
```

### Phase 1: READ_FILES Stage

#### 1.1 Extend SolverInterface Protocol

Add lifecycle methods to the solver protocol:

```python
# In foamadapter/framework/solver.py

class SolverInterface(Protocol):
    """Protocol defining the interface for solver implementations."""
    
    # Lifecycle Stage 1: READ_FILES
    def define_inputs(self) -> ModelInputCollection:
        """
        Define required input files and their validation models.
        
        Returns:
            ModelInputCollection with all required/optional input definitions
        """
        ...
    
    def read_files(self, case_dir: Path) -> None:
        """
        Read input files and initialize object state.
        
        Called during Stage 1 (READ_FILES).
        Should parse files and populate internal state.
        No inter-model dependencies at this stage.
        """
        ...
    
    # Lifecycle Stage 2: CONFIGURE
    def configure(self, models: dict[str, Any]) -> None:
        """
        Configure based on other models.
        
        Called during Stage 2 (CONFIGURE).
        Models can inspect other models and adjust their behavior.
        May modify which operations are returned by operations().
        
        Args:
            models: Dictionary of all registered models by name
        """
        ...
    
    # Lifecycle Stage 3: SETUP
    def setup(self, ctx: Context) -> None:
        """
        Prepare for execution before main loop.
        
        Called during Stage 3 (SETUP).
        Allocate resources, initialize matrices, pre-compute values.
        Context is fully available at this point.
        """
        ...
    
    # Runtime methods
    def operations(self, domain_name: str | None = None) -> OperationCollection:
        """Return collection of solver operations."""
        ...
    
    def main_loop(self, ctx: Context) -> None:
        """Execute main simulation loop."""
        ...
    
    def validate_configuration(self) -> list[ValidationErrors]:
        """
        Validate solver configuration (Pydantic model fields).
        
        Returns:
            List of ValidationErrors (empty if valid)
        """
        ...
```

#### 1.2 Implement Input Definition for IncompressibleFluid

```python
# In foamadapter/solver/incompressibleFluid.py

from foamadapter.io.input_validation import ModelInputDefinition, ModelInputCollection
from pybFoam.io.models import (  # Pydantic models for OpenFOAM files
    ControlDict,
    FvSchemes,
    FvSolution,
    TransportProperties,
)

@Solver
class IncompressibleFluid(BaseModel):
    # ... existing fields ...
    
    def define_inputs(self) -> ModelInputCollection:
        """
        Define required input files for incompressible solver.
        """
        inputs = ModelInputCollection()
        
        # System files
        inputs.add(ModelInputDefinition(
            baseModel=ControlDict,
            relative_path="system/controlDict",
            required=True,
            description="Time control and output settings"
        ))
        
        inputs.add(ModelInputDefinition(
            baseModel=FvSchemes,
            relative_path="system/fvSchemes",
            required=True,
            description="Discretization schemes"
        ))
        
        inputs.add(ModelInputDefinition(
            baseModel=FvSolution,
            relative_path="system/fvSolution",
            required=True,
            description="Linear solver and algorithm settings"
        ))
        
        # Constant files
        inputs.add(ModelInputDefinition(
            baseModel=TransportProperties,
            relative_path="constant/transportProperties",
            required=True,
            description="Physical properties (viscosity, etc.)"
        ))
        
        # Algorithm-specific inputs
        if self.algorithm == "PIMPLE":
            # Could add PIMPLE-specific validation here
            pass
        
        return inputs
    
    def validate_configuration(self) -> list[ValidationErrors]:
        """
        Validate solver configuration fields.
        """
        errors: list[ValidationErrors] = []
        
        # Check algorithm is supported
        if self.algorithm not in ["PIMPLE", "SIMPLE", "PISO"]:
            errors.append(ValidationErrors(
                field="algorithm",
                error_type="ValueError",
                message=f"Unsupported algorithm: {self.algorithm}",
                file_name="solver_config",
                input_value=self.algorithm
            ))
        
        # Check reference pressure settings
        if self.pRefCell is not None and self.pRefValue is None:
            errors.append(ValidationErrors(
                field="pRefValue",
                error_type="ValueError",
                message="pRefValue must be set when pRefCell is specified",
                file_name="solver_config",
                input_value=None
            ))
        
        # Check maxDeltaT is positive
        if self.maxDeltaT <= 0:
            errors.append(ValidationErrors(
                field="maxDeltaT",
                error_type="ValueError",
                message="maxDeltaT must be positive",
                file_name="solver_config",
                input_value=self.maxDeltaT
            ))
        
        return errors
```

#### 1.3 Validation Workflow

Add a validation function to run before solver initialization:

```python
# In foamadapter/framework/validation.py (NEW FILE)

from pathlib import Path
from typing import Tuple

from foamadapter.framework.solver import SolverInterface
from foamadapter.io.input_validation import ValidationErrors


def validate_solver(
    solver: SolverInterface,
    case_dir: str | Path = "."
) -> Tuple[bool, list[ValidationErrors]]:
    """
    Validate solver configuration and input files.
    
    Args:
        solver: Solver instance to validate
        case_dir: Path to OpenFOAM case directory
    
    Returns:
        Tuple of (is_valid, errors):
            - is_valid: True if all validations pass
            - errors: List of ValidationErrors (empty if valid)
    """
    all_errors: list[ValidationErrors] = []
    
    # Step 1: Validate solver configuration (Pydantic fields)
    config_errors = solver.validate_configuration()
    all_errors.extend(config_errors)
    
    # Step 2: Validate input files
    inputs = solver.define_inputs()
    is_valid, file_errors = inputs.validate_case(case_dir)
    all_errors.extend(file_errors)
    
    # Step 3: Check case directory structure
    case_path = Path(case_dir)
    required_dirs = ["system", "constant", "0"]
    for dir_name in required_dirs:
        if not (case_path / dir_name).exists():
            all_errors.append(ValidationErrors(
                field=None,
                error_type="DirectoryNotFound",
                message=f"Required directory not found: {dir_name}",
                file_name=str(case_path / dir_name),
            ))
    
    return len(all_errors) == 0, all_errors


def format_validation_errors(errors: list[ValidationErrors]) -> str:
    """
    Format validation errors into human-readable report.
    
    Returns:
        Multi-line string with error details
    """
    if not errors:
        return "✓ All validations passed"
    
    lines = [f"✗ Found {len(errors)} validation error(s):\n"]
    
    for i, error in enumerate(errors, 1):
        lines.append(f"{i}. [{error.error_type}] {error.message}")
        lines.append(f"   File: {error.file_name}")
        if error.field:
            lines.append(f"   Field: {error.field}")
        if error.input_value is not None:
            lines.append(f"   Input: {error.input_value}")
        lines.append("")
    
    return "\n".join(lines)
```

---

### Phase 2: Staged Initialization (Stages 2-3)

#### 2.1 Initialization Stages

Define explicit initialization stages:

```python
# In foamadapter/framework/initialization.py (NEW FILE)

from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from foamadapter.framework.context import Context
from foamadapter.framework.solver import SolverInterface


class InitializationStage(Enum):
    """Enumeration of solver initialization stages."""
    READ_FILES = auto()         # Stage 1: Read input files
    CONFIGURE = auto()          # Stage 2: Models configure based on other models
    SETUP = auto()              # Stage 3: Prepare for execution
    READY = auto()              # Ready to execute


@dataclass
class InitializationResult:
    """Result of an initialization stage."""
    success: bool
    stage: InitializationStage
    message: str = ""
    context: Context | None = None


class ModelRegistry:
    """
    Registry of all models participating in initialization.
    
    Models can query each other during CONFIGURE stage.
    """
    
    def __init__(self):
        self._models: dict[str, Any] = {}
        self._configured: set[str] = set()
    
    def register(self, name: str, model: Any) -> None:
        """Register a model by name."""
        self._models[name] = model
    
    def get(self, name: str) -> Any:
        """Get a model by name."""
        return self._models.get(name)
    
    def all_models(self) -> dict[str, Any]:
        """Return all registered models."""
        return dict(self._models)
    
    def mark_configured(self, name: str) -> None:
        """Mark a model as fully configured."""
        self._configured.add(name)
    
    def is_configured(self, name: str) -> bool:
        """Check if a model has been configured."""
        return name in self._configured


class SolverInitializer:
    """
    Manages 3-stage initialization of a solver.
    
    Stages:
        1. READ_FILES: Read input files, validate, create model objects
        2. CONFIGURE: Models configure based on other models (dynamic operations)
        3. SETUP: Prepare models for execution (allocate resources, pre-compute)
    
    Usage:
        initializer = SolverInitializer(solver, case_dir=".")
        result = initializer.initialize()
        if result.success:
            ctx = result.context
            solver.main_loop(ctx)
        else:
            print(result.message)
    """
    
    def __init__(
        self,
        solver: SolverInterface,
        case_dir: str = ".",
    ):
        self.solver = solver
        self.case_dir = Path(case_dir)
        self.current_stage: InitializationStage | None = None
        self.context: Context | None = None
        self.registry = ModelRegistry()
    
    def initialize(self) -> InitializationResult:
        """
        Run full 3-stage initialization sequence.
        
        Returns:
            InitializationResult with success status and context
        """
        # Stage 1: READ_FILES
        result = self._read_files()
        if not result.success:
            return result
        
        # Stage 2: CONFIGURE
        result = self._configure()
        if not result.success:
            return result
        
        # Stage 3: SETUP
        result = self._setup()
        if not result.success:
            return result
        
        return InitializationResult(
            success=True,
            stage=InitializationStage.READY,
            message="Solver initialized successfully",
            context=self.context
        )
    
    def _read_files(self) -> InitializationResult:
        """
        Stage 1: READ_FILES
        
        - Validate input files exist and are well-formed
        - Read files and create model objects
        - No inter-model dependencies at this stage
        """
        from foamadapter.framework.validation import validate_solver, format_validation_errors
        
        # First validate all inputs
        is_valid, errors = validate_solver(self.solver, self.case_dir)
        if not is_valid:
            error_report = format_validation_errors(errors)
            return InitializationResult(
                success=False,
                stage=InitializationStage.READ_FILES,
                message=f"Validation failed:\n{error_report}"
            )
        
        try:
            # Read files and initialize solver state
            self.solver.read_files(self.case_dir)
            
            # Register solver in model registry
            self.registry.register("solver", self.solver)
            
            # Create initial context (mesh, runtime, empty fields)
            self.context = self.solver.create_context()
            
            self.current_stage = InitializationStage.READ_FILES
            return InitializationResult(
                success=True,
                stage=InitializationStage.READ_FILES,
                message="Files read successfully"
            )
        except Exception as e:
            return InitializationResult(
                success=False,
                stage=InitializationStage.READ_FILES,
                message=f"Failed to read files: {e}"
            )
    
    def _configure(self) -> InitializationResult:
        """
        Stage 2: CONFIGURE
        
        - Models inspect other models and configure themselves
        - Dynamic operation selection (e.g., turbulence model picks operations)
        - Configuration can trigger reconfiguration of dependent models
        - Build dependency graph based on final operations
        """
        try:
            # Get all models from context
            all_models = self.registry.all_models()
            
            # Allow solver to configure based on other models
            self.solver.configure(all_models)
            self.registry.mark_configured("solver")
            
            # Configure algorithm (if present)
            if hasattr(self.solver, '_create_algorithm'):
                algorithm = self.solver._create_algorithm()
                self.registry.register("algorithm", algorithm)
                if hasattr(algorithm, 'configure'):
                    algorithm.configure(all_models)
                self.registry.mark_configured("algorithm")
            
            # Build and validate dependency graph
            ops = self.solver.operations()
            if hasattr(ops, 'is_acyclic') and not ops.is_acyclic():
                return InitializationResult(
                    success=False,
                    stage=InitializationStage.CONFIGURE,
                    message="Dependency graph contains cycles"
                )
            
            self.current_stage = InitializationStage.CONFIGURE
            return InitializationResult(
                success=True,
                stage=InitializationStage.CONFIGURE,
                message=f"Configuration complete, {len(ops)} operations registered"
            )
        except Exception as e:
            return InitializationResult(
                success=False,
                stage=InitializationStage.CONFIGURE,
                message=f"Configuration failed: {e}"
            )
    
    def _setup(self) -> InitializationResult:
        """
        Stage 3: SETUP
        
        - Allocate runtime resources
        - Initialize solver matrices and data structures
        - Perform pre-computation (geometric calculations, etc.)
        - Validate inter-model consistency
        """
        try:
            # Run setup on solver
            self.solver.setup(self.context)
            
            # Run setup on algorithm if present
            algorithm = self.registry.get("algorithm")
            if algorithm and hasattr(algorithm, 'setup'):
                algorithm.setup(self.context)
            
            self.current_stage = InitializationStage.SETUP
            return InitializationResult(
                success=True,
                stage=InitializationStage.SETUP,
                message="Setup complete, ready for execution",
                context=self.context
            )
        except Exception as e:
            return InitializationResult(
                success=False,
                stage=InitializationStage.SETUP,
                message=f"Setup failed: {e}"
            )
```

---

### Phase 3: Dynamic Configuration Examples

#### 3.1 Turbulence Model with Dynamic Operations

Example of a model that returns different operations based on configuration:

```python
# In foamadapter/models/turbulence.py

@Model
class TurbulenceModel(BaseModel):
    """
    Turbulence model with dynamic operation selection.
    
    During CONFIGURE stage, inspects solver settings to determine
    which operations to provide.
    """
    
    name: str = "kEpsilon"
    wall_functions: bool = True
    
    # Internal state set during configure
    _active_operations: list[str] = []
    
    def read_files(self, case_dir: Path) -> None:
        """Stage 1: Read turbulence properties."""
        # Read turbulenceProperties file
        props = read_turbulence_properties(case_dir)
        self.name = props.get("model", "kEpsilon")
        self.wall_functions = props.get("wallFunctions", True)
    
    def configure(self, models: dict[str, Any]) -> None:
        """
        Stage 2: Configure based on solver and other models.
        
        Decides which operations to provide based on:
        - Solver type (incompressible vs compressible)
        - Wall function settings
        - Other model configurations
        """
        solver = models.get("solver")
        
        # Base operations always included
        self._active_operations = ["correct_turbulence"]
        
        # Add wall function operations if enabled
        if self.wall_functions:
            self._active_operations.append("update_wall_functions")
        
        # Add compressibility corrections for compressible solvers
        if hasattr(solver, 'compressible') and solver.compressible:
            self._active_operations.append("compressibility_correction")
    
    def setup(self, ctx: Context) -> None:
        """Stage 3: Allocate turbulence fields."""
        mesh = ctx.mesh
        
        # Create k and epsilon fields
        k = volScalarField.read_field(mesh, "k")
        epsilon = volScalarField.read_field(mesh, "epsilon")
        
        ctx.fields["k"] = k
        ctx.fields["epsilon"] = epsilon
    
    def operations(self) -> OperationCollection:
        """Return only the configured operations."""
        ops = OperationCollection()
        
        for op_name in self._active_operations:
            method = getattr(self, op_name)
            ops.add(Operation.create_SeqOp(method))
        
        return ops
    
    @Model.operation()
    def correct_turbulence(self, turbulence) -> FieldUpdates:
        """Always-present turbulence correction."""
        turbulence.correct()
        return FieldUpdates({"turbulence": turbulence})
    
    @Model.operation()
    def update_wall_functions(self, k, epsilon) -> FieldUpdates:
        """Only included if wall_functions=True."""
        # Wall function updates...
        return FieldUpdates({"k": k, "epsilon": epsilon})
    
    @Model.operation()
    def compressibility_correction(self, rho, turbulence) -> FieldUpdates:
        """Only included for compressible solvers."""
        # Compressibility correction...
        return FieldUpdates({"turbulence": turbulence})
```

#### 3.2 Algorithm Configuration Based on Solver

```python
# In foamadapter/algorithms/pressure_velocity.py

@Model
class PimpleAlgorithm(BaseModel):
    """
    PIMPLE algorithm with configurable behavior.
    """
    
    nOuterCorrectors: int = 1
    nCorrectors: int = 2
    
    # Set during configure
    _use_momentum_predictor: bool = True
    _use_non_orthogonal_correction: bool = False
    
    def configure(self, models: dict[str, Any]) -> None:
        """
        Stage 2: Configure based on solver and mesh.
        
        - Check if mesh is orthogonal (skip non-orthogonal correction)
        - Check if transient (use momentum predictor)
        """
        solver = models.get("solver")
        
        # Transient solvers use momentum predictor
        self._use_momentum_predictor = getattr(solver, 'transient', True)
        
        # Could check mesh orthogonality here
        # self._use_non_orthogonal_correction = not mesh.is_orthogonal()
    
    def operations(self) -> OperationCollection:
        """Return operations based on configuration."""
        ops = OperationCollection()
        
        if self._use_momentum_predictor:
            ops.add(Operation.create_SeqOp(self.momentum))
        
        ops.add(Operation.create_SeqOp(self.continuity))
        
        return ops
```

---

### Phase 4: Integration with Existing Code

#### 4.1 Update Solver Main Entry Point

Modify how solvers are launched to include the 3-stage initialization:

```python
# Example: applications/incompressibleFluid/main.py

from foamadapter.solver.incompressibleFluid import IncompressibleFluid
from foamadapter.framework.initialization import SolverInitializer
from foamadapter.solver.incompressibleFluid import CFLCondition
import sys


def main():
    """Main entry point for incompressible fluid solver."""
    
    # Create solver configuration
    solver = IncompressibleFluid(
        argv=sys.argv,
        algorithm="PIMPLE",
    )
    
    # Initialize with validation
    initializer = SolverInitializer(solver, case_dir=".")
    result = initializer.initialize()
    
    if not result.success:
        print(result.message, file=sys.stderr)
        sys.exit(1)
    
    # Run simulation
    ctx = result.context
    condition = CFLCondition(solver.maxDeltaT)
    
    while condition(ctx):
        solver.main_loop(ctx)
    
    print("Simulation completed successfully")


if __name__ == "__main__":
    main()
```

#### 4.2 Backward Compatibility

For existing code that doesn't use validation, provide a compatibility path:

```python
# In foamadapter/framework/solver.py

def run_solver_with_validation(
    solver: SolverInterface,
    case_dir: str = ".",
    skip_validation: bool = False
) -> bool:
    """
    Run solver with optional validation.
    
    Args:
        solver: Solver instance
        case_dir: Case directory path
        skip_validation: If True, skip validation (not recommended)
    
    Returns:
        True if successful, False otherwise
    """
    if not skip_validation:
        from foamadapter.framework.initialization import SolverInitializer
        initializer = SolverInitializer(solver, case_dir)
        result = initializer.initialize()
        
        if not result.success:
            print(result.message, file=sys.stderr)
            return False
        
        ctx = result.context
    else:
        # Legacy path: direct context creation
        ctx = solver.create_context()
    
    # Run main loop (implementation depends on solver)
    solver.main_loop(ctx)
    return True
```

---

## Implementation Checklist

### Phase 1: READ_FILES Stage
- [ ] Add `define_inputs()` to `SolverInterface` protocol
- [ ] Add `read_files(case_dir)` to `SolverInterface` protocol
- [ ] Add `validate_configuration()` to `SolverInterface` protocol
- [ ] Implement `define_inputs()` in `IncompressibleFluid`
- [ ] Implement `read_files()` in `IncompressibleFluid`
- [ ] Implement `validate_configuration()` in `IncompressibleFluid`
- [ ] Create `foamadapter/framework/validation.py` with `validate_solver()` function
- [ ] Create Pydantic models for OpenFOAM files (ControlDict, FvSchemes, etc.) in `pybFoam`
- [ ] Add tests for file reading in `test/framework/test_read_files.py`

### Phase 2: CONFIGURE Stage
- [ ] Add `configure(models)` to `SolverInterface` protocol
- [ ] Implement `configure()` in `IncompressibleFluid`
- [ ] Implement `ModelRegistry` for inter-model communication
- [ ] Add `configure()` to `PimpleAlgorithm`
- [ ] Test dynamic operation selection based on configuration
- [ ] Validate that `operations()` can return different results after configure

### Phase 3: SETUP Stage
- [ ] Add `setup(ctx)` to `SolverInterface` protocol
- [ ] Implement `setup()` in `IncompressibleFluid`
- [ ] Add `setup()` to `PimpleAlgorithm`
- [ ] Test resource allocation during setup
- [ ] Test pre-computation (geometric calculations, etc.)

### Phase 4: SolverInitializer
- [ ] Create `foamadapter/framework/initialization.py`
- [ ] Implement `SolverInitializer` class with 3-stage initialization
- [ ] Add `InitializationStage` enum and `InitializationResult` dataclass
- [ ] Add tests for `SolverInitializer` in `test/framework/test_initialization.py`
- [ ] Test error handling for each stage

### Phase 5: Integration & Testing
- [ ] Update solver applications to use `SolverInitializer`
- [ ] Add `run_solver_with_validation()` convenience function
- [ ] Update documentation with 3-stage initialization workflow
- [ ] Add integration tests for full initialization workflow
- [ ] Ensure backward compatibility with existing code

---

## Benefits

1. **Clear Separation of Concerns**: Each stage has a specific purpose
   - READ_FILES: Parse and validate inputs (no dependencies)
   - CONFIGURE: Dynamic behavior based on other models
   - SETUP: Allocate resources, prepare for execution
2. **Early Error Detection**: Catch configuration/input errors before attempting to run
3. **Dynamic Operation Selection**: Models can return different operations based on configuration
4. **Inter-Model Communication**: Models can query and react to each other during CONFIGURE
5. **Better Error Messages**: Structured errors with file names, field names, and suggestions
6. **Fail Fast**: Don't waste time on long initialization if inputs are invalid
7. **Type Safety**: Pydantic models ensure all inputs match expected schema
8. **Extensibility**: Easy to add new models with lifecycle hooks
9. **Testability**: Each stage can be tested independently

---

## Example Usage

### Valid Case

```python
solver = IncompressibleFluid(argv=sys.argv, algorithm="PIMPLE")
initializer = SolverInitializer(solver, case_dir="./cavity")
result = initializer.initialize()

if result.success:
    print("✓ Solver ready")
    solver.main_loop(result.context)
else:
    print(result.message)
```

**Output:**
```
✓ Solver ready
Time = 0.001
Time = 0.002
...
```

### Invalid Case (Missing File)

```python
solver = IncompressibleFluid(argv=sys.argv, algorithm="PIMPLE")
initializer = SolverInitializer(solver, case_dir="./bad_case")
result = initializer.initialize()
```

**Output:**
```
✗ Found 2 validation error(s):

1. [FileNotFound] File not found
   File: ./bad_case/system/controlDict

2. [FileNotFound] File not found
   File: ./bad_case/constant/transportProperties
```

### Invalid Configuration

```python
solver = IncompressibleFluid(
    argv=sys.argv,
    algorithm="INVALID",  # Not supported
    maxDeltaT=-1.0        # Must be positive
)
initializer = SolverInitializer(solver)
result = initializer.initialize()
```

**Output:**
```
✗ Found 2 validation error(s):

1. [ValueError] Unsupported algorithm: INVALID
   File: solver_config
   Field: algorithm
   Input: INVALID

2. [ValueError] maxDeltaT must be positive
   File: solver_config
   Field: maxDeltaT
   Input: -1.0
```

---

## Future Enhancements

1. **Validation Hooks**: Allow users to add custom validation logic
2. **Incremental Validation**: Validate only changed files
3. **Schema Export**: Generate JSON schemas for documentation
4. **IDE Integration**: Provide autocomplete for configuration fields
5. **Validation Warnings**: Non-fatal warnings for suspicious but valid inputs
6. **Parallel Validation**: Validate multiple files concurrently
7. **Caching**: Cache validation results to speed up repeated runs

---

## Summary

This plan extends NeoFOAM's solver framework with:

1. **Input validation** using existing `ModelInputCollection` infrastructure
2. **Configuration validation** via Pydantic model checks
3. **Staged initialization** with clear success/failure reporting
4. **Better error messages** showing exactly what's wrong and where
5. **Backward compatibility** for existing code

The implementation leverages existing Pydantic and validation infrastructure, requiring only:
- Protocol extensions (`define_inputs()`, `validate_configuration()`)
- New initialization orchestration (`SolverInitializer`)
- Pydantic models for OpenFOAM file formats

This provides a robust foundation for reliable solver initialization with excellent user experience.
