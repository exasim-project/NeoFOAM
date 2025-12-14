# Implementation: 3-Stage Initialization with TestSolver

## TestSolver Implementation

The TestSolver demonstrates the complete 3-stage initialization architecture with multiple test models.

### Test Models

```python
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel

# ============================================================================
# Stage Decorators
# ============================================================================

def read_files(func):
    """Mark a method as belonging to READ_FILES stage."""
    func._init_stage = "READ_FILES"
    return func

def configure(func):
    """Mark a method as belonging to CONFIGURE stage."""
    func._init_stage = "CONFIGURE"
    return func

def setup(func):
    """Mark a method as belonging to SETUP stage."""
    func._init_stage = "SETUP"
    return func

# ============================================================================
# Configuration Schemas
# ============================================================================

class TurbulenceConfig(BaseModel):
    """Configuration for turbulence model."""
    model_type: str = "kEpsilon"
    wall_function: bool = True
    coefficients: dict = {}

class TransportConfig(BaseModel):
    """Configuration for transport properties."""
    viscosity: float = 1e-6
    density: float = 1000.0

class SolverConfig(BaseModel):
    """Main solver configuration."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    time_step: float = 0.001

# ============================================================================
# Test Models
# ============================================================================

@dataclass
class TestTurbulenceModel:
    """Test turbulence model with 3-stage initialization."""
    
    name: str = "turbulence"
    config: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    
    # State tracking for testing
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False
    transport_ref: Any = None  # Reference from CONFIGURE stage
    
    @read_files
    def load_coefficients(self):
        """READ_FILES: Load turbulence coefficients from file."""
        # Simulate reading from file
        self.config.coefficients = {"C_mu": 0.09, "C_1": 1.44, "C_2": 1.92}
        self.files_read = True
        return self.config.coefficients
    
    @configure
    def connect_transport(self, registry: "ModelRegistry"):
        """CONFIGURE: Connect to transport model for viscosity."""
        # Get reference to transport model
        transport = registry.get("transport")
        if transport:
            self.transport_ref = transport
            self.configured = True
        return self.configured
    
    @setup
    def initialize_fields(self, mesh):
        """SETUP: Initialize turbulence fields on mesh."""
        # Create k and epsilon fields
        if self.transport_ref:
            nu = self.transport_ref.config.viscosity
            # Use viscosity for initial turbulence estimates
            self.setup_complete = True
        return self.setup_complete


@dataclass
class TestTransportModel:
    """Test transport properties model."""
    
    name: str = "transport"
    config: TransportConfig = field(default_factory=TransportConfig)
    
    # State tracking
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False
    
    @read_files
    def load_properties(self):
        """READ_FILES: Load transport properties from file."""
        # Simulate reading transportProperties
        self.config.viscosity = 1e-6
        self.config.density = 998.0
        self.files_read = True
        return self.config
    
    @configure
    def validate_properties(self, registry: "ModelRegistry"):
        """CONFIGURE: Validate transport properties."""
        if self.config.viscosity <= 0:
            raise ValueError("Viscosity must be positive")
        if self.config.density <= 0:
            raise ValueError("Density must be positive")
        self.configured = True
        return self.configured
    
    @setup
    def create_fields(self, mesh):
        """SETUP: Create transport coefficient fields."""
        # Create nu and rho fields on mesh
        self.setup_complete = True
        return self.setup_complete


@dataclass  
class TestAlgorithmModel:
    """Test pressure-velocity coupling algorithm."""
    
    name: str = "algorithm"
    config: dict = field(default_factory=dict)
    
    # State tracking
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False
    turbulence_ref: Any = None
    transport_ref: Any = None
    
    @read_files
    def load_settings(self):
        """READ_FILES: Load algorithm settings."""
        self.config = {
            "n_correctors": 2,
            "n_outer_correctors": 1,
            "momentum_predictor": True
        }
        self.files_read = True
        return self.config
    
    @configure
    def connect_models(self, registry: "ModelRegistry"):
        """CONFIGURE: Connect to turbulence and transport models."""
        self.turbulence_ref = registry.get("turbulence")
        self.transport_ref = registry.get("transport")
        self.configured = True
        return self.configured
    
    @setup
    def setup_matrices(self, mesh):
        """SETUP: Set up matrix systems."""
        self.setup_complete = True
        return self.setup_complete

# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Central registry for inter-model communication during CONFIGURE."""
    
    def __init__(self):
        self._models: dict[str, Any] = {}
    
    def register(self, name: str, model: Any):
        """Register a model by name."""
        self._models[name] = model
    
    def get(self, name: str) -> Any:
        """Get a registered model by name."""
        return self._models.get(name)
    
    def all(self) -> dict[str, Any]:
        """Get all registered models."""
        return self._models.copy()

# ============================================================================
# TestSolver
# ============================================================================

@dataclass
class TestSolver:
    """Test solver with 3-stage initialization and multiple models."""
    
    config: SolverConfig = field(default_factory=SolverConfig)
    
    # Models (1 solver with N models)
    turbulence: TestTurbulenceModel = field(default_factory=TestTurbulenceModel)
    transport: TestTransportModel = field(default_factory=TestTransportModel)
    algorithm: TestAlgorithmModel = field(default_factory=TestAlgorithmModel)
    
    # Solver state
    files_read: bool = False
    configured: bool = False
    setup_complete: bool = False
    
    # Internal
    _registry: ModelRegistry = field(default_factory=ModelRegistry)
    
    def get_models(self) -> list:
        """Return all models owned by this solver."""
        return [self.turbulence, self.transport, self.algorithm]
    
    # ---- Solver's own lifecycle methods ----
    
    @read_files
    def load_control_dict(self):
        """READ_FILES: Load solver control settings."""
        self.config.max_iterations = 100
        self.config.tolerance = 1e-6
        self.files_read = True
        return self.config
    
    @configure
    def validate_config(self, registry: ModelRegistry):
        """CONFIGURE: Validate solver configuration."""
        # Check all models are configured
        for model in self.get_models():
            if not model.configured:
                raise RuntimeError(f"Model {model.name} not configured")
        self.configured = True
        return self.configured
    
    @setup
    def create_solver_context(self, mesh):
        """SETUP: Create solver execution context."""
        self.setup_complete = True
        return self.setup_complete


# ============================================================================
# Solver Initializer
# ============================================================================

class SolverInitializer:
    """Orchestrates 3-stage initialization for solver and its models."""
    
    def __init__(self, solver):
        self.solver = solver
        self.registry = ModelRegistry()
        
    def initialize(self, mesh=None):
        """Run complete 3-stage initialization."""
        self._run_read_files()
        self._run_configure()
        self._run_setup(mesh)
        return self.solver
    
    def _run_read_files(self):
        """Execute READ_FILES stage on solver and all models."""
        # Models first
        for model in self.solver.get_models():
            self._execute_stage_methods(model, "READ_FILES")
            self.registry.register(model.name, model)
        
        # Then solver
        self._execute_stage_methods(self.solver, "READ_FILES")
    
    def _run_configure(self):
        """Execute CONFIGURE stage - models can reference each other."""
        # Models first (they may depend on each other)
        for model in self.solver.get_models():
            self._execute_stage_methods(model, "CONFIGURE", self.registry)
        
        # Then solver (can validate all models are configured)
        self._execute_stage_methods(self.solver, "CONFIGURE", self.registry)
    
    def _run_setup(self, mesh):
        """Execute SETUP stage with mesh context."""
        # Models first
        for model in self.solver.get_models():
            self._execute_stage_methods(model, "SETUP", mesh)
        
        # Then solver
        self._execute_stage_methods(self.solver, "SETUP", mesh)
    
    def _execute_stage_methods(self, obj, stage: str, *args):
        """Execute all methods marked with given stage decorator."""
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            attr = getattr(obj, attr_name, None)
            if callable(attr) and getattr(attr, "_init_stage", None) == stage:
                attr(*args)
```

## Tests

```python
import pytest

class TestThreeStageInitialization:
    """Comprehensive tests for 3-stage initialization architecture."""
    
    # ========================================================================
    # Basic Initialization Tests
    # ========================================================================
    
    def test_solver_creation(self):
        """Test that solver can be created with default models."""
        solver = TestSolver()
        
        assert solver.turbulence is not None
        assert solver.transport is not None
        assert solver.algorithm is not None
    
    def test_get_models_returns_all_models(self):
        """Test that get_models returns all embedded models."""
        solver = TestSolver()
        models = solver.get_models()
        
        assert len(models) == 3
        assert solver.turbulence in models
        assert solver.transport in models
        assert solver.algorithm in models
    
    # ========================================================================
    # READ_FILES Stage Tests
    # ========================================================================
    
    def test_read_files_stage_marks_methods(self):
        """Test that @read_files decorator marks methods correctly."""
        model = TestTurbulenceModel()
        
        assert hasattr(model.load_coefficients, "_init_stage")
        assert model.load_coefficients._init_stage == "READ_FILES"
    
    def test_read_files_stage_executes_all_models(self):
        """Test that READ_FILES executes on all models."""
        solver = TestSolver()
        initializer = SolverInitializer(solver)
        
        initializer._run_read_files()
        
        assert solver.turbulence.files_read
        assert solver.transport.files_read
        assert solver.algorithm.files_read
        assert solver.files_read
    
    def test_read_files_loads_correct_data(self):
        """Test that READ_FILES loads expected data."""
        solver = TestSolver()
        initializer = SolverInitializer(solver)
        
        initializer._run_read_files()
        
        # Check turbulence loaded coefficients
        assert "C_mu" in solver.turbulence.config.coefficients
        assert solver.turbulence.config.coefficients["C_mu"] == 0.09
        
        # Check transport loaded properties
        assert solver.transport.config.viscosity == 1e-6
        assert solver.transport.config.density == 998.0
    
    # ========================================================================
    # CONFIGURE Stage Tests
    # ========================================================================
    
    def test_configure_stage_marks_methods(self):
        """Test that @configure decorator marks methods correctly."""
        model = TestTurbulenceModel()
        
        assert hasattr(model.connect_transport, "_init_stage")
        assert model.connect_transport._init_stage == "CONFIGURE"
    
    def test_configure_stage_registers_models(self):
        """Test that models are registered in registry during CONFIGURE."""
        solver = TestSolver()
        initializer = SolverInitializer(solver)
        
        initializer._run_read_files()
        
        # After read_files, models should be registered
        assert initializer.registry.get("turbulence") is solver.turbulence
        assert initializer.registry.get("transport") is solver.transport
        assert initializer.registry.get("algorithm") is solver.algorithm
    
    def test_configure_stage_connects_models(self):
        """Test that models can reference each other during CONFIGURE."""
        solver = TestSolver()
        initializer = SolverInitializer(solver)
        
        initializer._run_read_files()
        initializer._run_configure()
        
        # Turbulence should have reference to transport
        assert solver.turbulence.transport_ref is solver.transport
        
        # Algorithm should have references to both
        assert solver.algorithm.turbulence_ref is solver.turbulence
        assert solver.algorithm.transport_ref is solver.transport
    
    def test_configure_stage_marks_models_configured(self):
        """Test that CONFIGURE marks all models as configured."""
        solver = TestSolver()
        initializer = SolverInitializer(solver)
        
        initializer._run_read_files()
        initializer._run_configure()
        
        assert solver.turbulence.configured
        assert solver.transport.configured
        assert solver.algorithm.configured
        assert solver.configured
    
    # ========================================================================
    # SETUP Stage Tests
    # ========================================================================
    
    def test_setup_stage_marks_methods(self):
        """Test that @setup decorator marks methods correctly."""
        model = TestTurbulenceModel()
        
        assert hasattr(model.initialize_fields, "_init_stage")
        assert model.initialize_fields._init_stage == "SETUP"
    
    def test_setup_stage_completes_initialization(self):
        """Test that SETUP completes all model initialization."""
        solver = TestSolver()
        initializer = SolverInitializer(solver)
        
        initializer._run_read_files()
        initializer._run_configure()
        initializer._run_setup(mesh=None)
        
        assert solver.turbulence.setup_complete
        assert solver.transport.setup_complete
        assert solver.algorithm.setup_complete
        assert solver.setup_complete
    
    # ========================================================================
    # Full Initialization Tests
    # ========================================================================
    
    def test_full_initialization(self):
        """Test complete 3-stage initialization flow."""
        solver = TestSolver()
        initializer = SolverInitializer(solver)
        
        result = initializer.initialize(mesh=None)
        
        # All stages complete
        assert solver.files_read
        assert solver.configured
        assert solver.setup_complete
        
        # All models initialized
        for model in solver.get_models():
            assert model.files_read
            assert model.configured
            assert model.setup_complete
        
        # Return solver
        assert result is solver
    
    def test_initialization_order(self):
        """Test that stages execute in correct order."""
        order = []
        
        class TrackingModel:
            name = "tracking"
            files_read = False
            configured = False
            setup_complete = False
            
            @read_files
            def read(self):
                order.append("READ_FILES")
                self.files_read = True
            
            @configure
            def config(self, registry):
                order.append("CONFIGURE")
                self.configured = True
            
            @setup  
            def set(self, mesh):
                order.append("SETUP")
                self.setup_complete = True
        
        class TrackingSolver:
            files_read = False
            configured = False
            setup_complete = False
            model = TrackingModel()
            
            def get_models(self):
                return [self.model]
            
            @read_files
            def read(self):
                order.append("SOLVER_READ")
                self.files_read = True
            
            @configure
            def config(self, registry):
                order.append("SOLVER_CONFIGURE")
                self.configured = True
            
            @setup
            def set(self, mesh):
                order.append("SOLVER_SETUP")
                self.setup_complete = True
        
        solver = TrackingSolver()
        initializer = SolverInitializer(solver)
        initializer.initialize(mesh=None)
        
        # Models before solver, stages in order
        expected = [
            "READ_FILES", "SOLVER_READ",
            "CONFIGURE", "SOLVER_CONFIGURE", 
            "SETUP", "SOLVER_SETUP"
        ]
        assert order == expected
    
    # ========================================================================
    # Error Handling Tests
    # ========================================================================
    
    def test_configure_fails_if_model_not_found(self):
        """Test behavior when model dependency is missing."""
        class OrphanModel:
            name = "orphan"
            files_read = True
            configured = False
            setup_complete = False
            dependency_found = False
            
            @configure
            def check_dependency(self, registry):
                missing = registry.get("nonexistent")
                self.dependency_found = missing is not None
                self.configured = True
        
        model = OrphanModel()
        registry = ModelRegistry()
        registry.register("orphan", model)
        
        model.check_dependency(registry)
        
        assert model.configured
        assert not model.dependency_found
    
    def test_validation_error_in_configure(self):
        """Test that validation errors are raised during CONFIGURE."""
        class InvalidTransport:
            name = "transport"
            files_read = True
            configured = False
            setup_complete = False
            config = TransportConfig(viscosity=-1.0)  # Invalid!
            
            @configure
            def validate(self, registry):
                if self.config.viscosity <= 0:
                    raise ValueError("Viscosity must be positive")
                self.configured = True
        
        model = InvalidTransport()
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="Viscosity must be positive"):
            model.validate(registry)
    
    # ========================================================================
    # Model Registry Tests
    # ========================================================================
    
    def test_registry_register_and_get(self):
        """Test basic registry operations."""
        registry = ModelRegistry()
        model = TestTurbulenceModel()
        
        registry.register("turb", model)
        
        assert registry.get("turb") is model
        assert registry.get("unknown") is None
    
    def test_registry_all(self):
        """Test getting all models from registry."""
        registry = ModelRegistry()
        turb = TestTurbulenceModel()
        trans = TestTransportModel()
        
        registry.register("turbulence", turb)
        registry.register("transport", trans)
        
        all_models = registry.all()
        
        assert len(all_models) == 2
        assert all_models["turbulence"] is turb
        assert all_models["transport"] is trans
    
    # ========================================================================
    # Configuration Validation Tests
    # ========================================================================
    
    def test_pydantic_config_validation(self):
        """Test that Pydantic validates configuration."""
        # Valid config
        config = TurbulenceConfig(model_type="kOmega", wall_function=False)
        assert config.model_type == "kOmega"
        
        # Default values
        default_config = TurbulenceConfig()
        assert default_config.model_type == "kEpsilon"
        assert default_config.wall_function is True
    
    def test_solver_config_defaults(self):
        """Test solver configuration defaults."""
        config = SolverConfig()
        
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
        assert config.time_step == 0.001
    
    # ========================================================================
    # Decorator Tests
    # ========================================================================
    
    def test_multiple_methods_same_stage(self):
        """Test that multiple methods can have same stage decorator."""
        class MultiMethodModel:
            name = "multi"
            files_read = False
            configured = True
            setup_complete = True
            data1 = None
            data2 = None
            
            @read_files
            def load_data1(self):
                self.data1 = "loaded1"
            
            @read_files
            def load_data2(self):
                self.data2 = "loaded2"
                self.files_read = True
        
        class MultiSolver:
            files_read = True
            configured = True
            setup_complete = True
            model = MultiMethodModel()
            
            def get_models(self):
                return [self.model]
        
        solver = MultiSolver()
        initializer = SolverInitializer(solver)
        initializer._run_read_files()
        
        assert solver.model.data1 == "loaded1"
        assert solver.model.data2 == "loaded2"
    
    def test_method_without_decorator_not_called(self):
        """Test that methods without stage decorators are not called."""
        class SelectiveModel:
            name = "selective"
            files_read = True
            configured = True
            setup_complete = True
            decorated_called = False
            undecorated_called = False
            
            @read_files
            def decorated_method(self):
                self.decorated_called = True
            
            def undecorated_method(self):
                self.undecorated_called = True
        
        class SelectiveSolver:
            files_read = True
            configured = True
            setup_complete = True
            model = SelectiveModel()
            
            def get_models(self):
                return [self.model]
        
        solver = SelectiveSolver()
        initializer = SolverInitializer(solver)
        initializer._run_read_files()
        
        assert solver.model.decorated_called
        assert not solver.model.undecorated_called
```

## Usage Example

```python
# Create solver with all models
solver = TestSolver(
    config=SolverConfig(max_iterations=200, tolerance=1e-8),
    turbulence=TestTurbulenceModel(),
    transport=TestTransportModel(),
    algorithm=TestAlgorithmModel()
)

# Initialize through 3-stage process
initializer = SolverInitializer(solver)
solver = initializer.initialize(mesh=my_mesh)

# Solver and all models are now fully initialized
assert solver.setup_complete
assert all(m.setup_complete for m in solver.get_models())
```

## Key Design Decisions

1. **1 Solver with N Models**: The solver owns its models as attributes, `get_models()` returns them.

2. **Stage Decorators**: `@read_files`, `@configure`, `@setup` mark which stage a method belongs to.

3. **ModelRegistry**: Allows models to find and reference each other during CONFIGURE stage.

4. **Initialization Order**: Models initialize before solver in each stage.

5. **Testable State**: Each model/solver tracks its state (files_read, configured, setup_complete) for testing.
