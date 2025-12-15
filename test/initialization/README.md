# 3-Stage Initialization Tests

This directory contains organized tests for the 3-stage initialization framework.

## Test Structure

The tests are split into focused modules:

### `test_fixtures.py`
Shared test fixtures including:
- Configuration schemas (TurbulenceConfig, TransportConfig, etc.)
- Test models (TestTurbulenceModel, TestTransportModel, TestAlgorithmModel)
- TestSolver implementation

### `test_basic.py` (3 tests)
Basic initialization functionality:
- Solver creation
- Model retrieval
- Full initialization flow

### `test_load_stage.py` (3 tests)
LOAD stage tests:
- Decorator marking
- Stage execution
- Data loading

### `test_resolve_dependencies_stage.py` (4 tests)
RESOLVE_DEPENDENCIES stage tests:
- Decorator marking
- Model registration in ConfigContext
- Inter-model connections
- Configuration validation

### `test_build_stage.py` (2 tests)
BUILD stage tests:
- Decorator marking
- Initialization completion

### `test_lazy_init.py` (14 tests)
LazyInit dataclass and helper function tests:
- LazyInit creation and execution
- Helper functions (field, operator, lazy, model)
- Dependency specification
- Context passing

### `test_lazy_build_integration.py` (7 tests)
Lazy BUILD stage integration tests:
- Field dependencies
- Operator dependencies
- Cycle detection
- DAG-based execution ordering
- Context passing to initializers

### `test_incompressible_fluid.py` (25 tests)
Full IncompressibleFluid solver initialization tests:
- Solver creation and configuration
- LOAD stage execution
- RESOLVE_DEPENDENCIES stage execution
- BUILD stage with lazy initialization
- Full initialization flow
- Context creation

### `test_initialization_order.py` (1 test)
Tests that stages execute in correct order:
- LOAD → RESOLVE_DEPENDENCIES → BUILD
- Models before solver in each stage

### `test_error_handling.py` (2 tests)
Error handling tests:
- Missing model dependencies
- Validation errors

### `test_config_context.py` (5 tests)
ConfigContext functionality:
- Registration and retrieval
- Getting all models
- Contains check
- Type-based queries
- Prefix-based queries

### `test_config_validation.py` (3 tests)
Pydantic configuration validation:
- Configuration validation
- Default values
- Constraint validation

### `test_configurable_field.py` (11 tests)
Configurable field behavior:
- Field marking and discovery
- Dynamic model adaptation
- Multiple adaptable fields
- Validation

### `test_decorators.py` (2 tests)
Decorator behavior:
- Multiple methods with same decorator
- Non-decorated methods not called

## Running Tests

Run all initialization tests:
```bash
pytest test/initialization/ -v
```

Run specific test file:
```bash
pytest test/initialization/test_basic.py -v
```

Run with coverage:
```bash
pytest test/initialization/ --cov=foamadapter.framework.initialization
```

## Key Features

### Lazy Initialization
The BUILD stage now supports lazy initialization where `@Solver.build` methods return `list[LazyInit]` objects instead of executing immediately. This enables:
- **Explicit dependencies**: Each initialization step declares what it depends on
- **Automatic ordering**: DAG-based topological sort determines execution order
- **Cycle detection**: Circular dependencies caught before execution
- **Better testability**: Individual initialization steps can be tested in isolation

### Helper Functions
Convenience functions for creating LazyInit objects:
- `field(name, initializer, depends_on)` - Creates fields with name prefix "fields."
- `operator(name, initializer, depends_on)` - Creates operators with name prefix "operators."
- `model(name, initializer, depends_on)` - Creates models with name prefix "models."
- `lazy(name, initializer, depends_on)` - Creates LazyInit with custom name

## Total: 82 initialization tests
