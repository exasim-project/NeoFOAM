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

### `test_read_files_stage.py` (3 tests)
READ_FILES stage tests:
- Decorator marking
- Stage execution
- Data loading

### `test_configure_stage.py` (4 tests)
CONFIGURE stage tests:
- Decorator marking
- Model registration in registry
- Inter-model connections
- Configuration validation

### `test_setup_stage.py` (2 tests)
SETUP stage tests:
- Decorator marking
- Initialization completion

### `test_initialization_order.py` (1 test)
Tests that stages execute in correct order:
- READ_FILES → CONFIGURE → SETUP
- Models before solver in each stage

### `test_error_handling.py` (2 tests)
Error handling tests:
- Missing model dependencies
- Validation errors

### `test_registry.py` (3 tests)
ModelRegistry functionality:
- Registration and retrieval
- Getting all models
- Contains check

### `test_config_validation.py` (3 tests)
Pydantic configuration validation:
- Configuration validation
- Default values
- Constraint validation

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

## Total: 23 tests
