# Implementation Plan: ConfigContext and Configurable Fields

This document outlines the implementation plan to replace `ModelRegistry` with `ConfigContext` and `AdaptableField` with `Configurable[T]` type annotations.

---

## Summary

| Current | New | Change |
|---------|-----|--------|
| `ModelRegistry` | `ConfigContext` | Renamed, supports dot notation for cross-region access |
| `AdaptableField(...)` | `Configurable[T]` | Type annotation instead of field wrapper |
| `registry.get("model")` | `config.get("model")` | Same region access |
| N/A | `config.get("region.model")` | Cross-region access via dot notation |

---

## Changed Files

| File | Classes/Functions | Description | Reason |
|------|-------------------|-------------|--------|
| `src/foamadapter/framework/initialization/registry.py` | `ModelRegistry` → `ConfigContext` | Rename class, add dot notation parsing for cross-region access | Better naming, multi-region support |
| `src/foamadapter/framework/initialization/adaptable_field.py` | `AdaptableField` → `Configurable` | Replace with `Annotated` type alias | Type shows intent, cleaner syntax |
| `src/foamadapter/framework/initialization/__init__.py` | Exports | Update exports: `ConfigContext`, `Configurable` | Public API change |
| `src/foamadapter/framework/initialization/initializer.py` | `SolverInitializer` | Use `ConfigContext`, add region support | Multi-region initialization |
| `src/foamadapter/framework/initialization/decorators.py` | Docstrings | Update `registry` → `config` in docs | Consistency |
| `src/foamadapter/framework/__init__.py` | Exports | Update exports | Public API change |
| `src/foamadapter/framework/model.py` | `Model` | Update docstrings and type hints | Consistency |
| `src/foamadapter/framework/solver.py` | `Solver` | Update docstrings and type hints | Consistency |
| `src/foamadapter/framework/model_protocol.py` | Protocol docs | Update `ModelRegistry` → `ConfigContext` | Consistency |
| `src/foamadapter/solver/incompressibleFluid.py` | `IncompressibleFluid` | Update `resolve_dependencies` signature | Use new API |
| `test/initialization/test_registry.py` | All tests | Rename to `test_config_context.py`, update tests | Match new naming |
| `test/initialization/test_adaptable_field.py` | All tests | Update to use `Configurable[T]` | Match new API |
| `test/initialization/test_fixtures.py` | Fixture models | Update `ModelRegistry` → `ConfigContext`, `AdaptableField` → `Configurable` | Match new API |
| `test/initialization/test_incompressible_fluid.py` | All tests | Update `ModelRegistry` → `ConfigContext` | Match new API |
| `doc/development/initialization.rst` | Documentation | Update all references | Consistency |

---

## Implementation Steps

### Phase 1: Core Changes

#### Step 1.1: Create `Configurable` Type

**File:** `src/foamadapter/framework/initialization/configurable.py` (new file)

```python
from typing import Annotated, TypeVar

T = TypeVar('T')

# Type alias for configurable fields
# Usage: use_buoyancy: Configurable[bool] = False
Configurable = Annotated
```

#### Step 1.2: Rename `ModelRegistry` → `ConfigContext`

**File:** `src/foamadapter/framework/initialization/registry.py` → `config_context.py`

```python
class ConfigContext:
    """Context for inter-model configuration exchange."""

    def __init__(self, current_region: str = "default"):
        self.current_region = current_region
        self._regions: dict[str, dict[str, Any]] = {"default": {}}

    def register(self, name: str, model: Any, region: str = None) -> None:
        """Register a model."""
        region = region or self.current_region
        if region not in self._regions:
            self._regions[region] = {}
        self._regions[region][name] = model

    def get(self, path: str) -> Any:
        """
        Get model by path.

        - "model_name" → model in current region
        - "region.model_name" → model in specified region
        """
        if "." in path:
            region, name = path.split(".", 1)
        else:
            region = self.current_region
            name = path

        return self._regions.get(region, {}).get(name)

    # Keep existing methods: all(), contains(), get_by_type(), get_by_prefix()
```

#### Step 1.3: Update `__init__.py` Exports

**File:** `src/foamadapter/framework/initialization/__init__.py`

```python
from .config_context import ConfigContext
from .configurable import Configurable

# Deprecate but keep for transition (optional)
ModelRegistry = ConfigContext  # Alias
AdaptableField = ...  # Keep temporarily

__all__ = [
    "ConfigContext",
    "Configurable",
    # ... other exports
]
```

### Phase 2: Update Initializer

#### Step 2.1: Update `SolverInitializer`

**File:** `src/foamadapter/framework/initialization/initializer.py`

- Replace `ModelRegistry` with `ConfigContext`
- Add region parameter support
- Update `_run_resolve_dependencies` to pass `ConfigContext`

### Phase 3: Update Solvers and Models

#### Step 3.1: Update `IncompressibleFluid`

**File:** `src/foamadapter/solver/incompressibleFluid.py`

```python
# Before
from foamadapter.framework.initialization import ModelRegistry

def configure_solver(self, registry: ModelRegistry) -> None:
    ...

# After
from foamadapter.framework.initialization import ConfigContext

def configure_solver(self, config: ConfigContext) -> None:
    ...
```

### Phase 4: Update Tests

#### Step 4.1: Rename and Update Test Files

| Old File | New File |
|----------|----------|
| `test/initialization/test_registry.py` | `test/initialization/test_config_context.py` |
| `test/initialization/test_adaptable_field.py` | `test/initialization/test_configurable.py` |

#### Step 4.2: Update All Test Fixtures

**File:** `test/initialization/test_fixtures.py`

```python
# Before
from foamadapter.framework import AdaptableField, ModelRegistry

class MyModel(BaseModel):
    use_feature: bool = AdaptableField(default=False)

    def resolve(self, registry: ModelRegistry):
        other = registry.get("other")

# After
from foamadapter.framework import Configurable, ConfigContext

class MyModel(BaseModel):
    use_feature: Configurable[bool] = False

    def resolve(self, config: ConfigContext):
        other = config.get("other")
```

### Phase 5: Update Documentation

**File:** `doc/development/initialization.rst`

- Replace all `ModelRegistry` → `ConfigContext`
- Replace all `AdaptableField` → `Configurable`
- Update examples
- Add section on cross-region access with dot notation

---

## New API Examples

### Single-Region Usage

```python
from foamadapter.framework import Configurable, ConfigContext, Model

class VelocityModel(BaseModel):
    # Configurable field - type shows intent
    use_buoyancy: Configurable[bool] = False
    g: Configurable[tuple] = (0, 0, -9.81)

    # Regular field - not configurable
    relax: float = 0.7


class BuoyancyModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, config: ConfigContext):
        velocity = config.get("velocity")
        velocity.use_buoyancy = True
        velocity.g = (0, 0, -9.81)
```

### Multi-Region Usage

```python
class CHTCouplingModel(BaseModel):
    @Model.resolve_dependencies
    def resolve(self, config: ConfigContext):
        # Cross-region access via dot notation
        fluid_temp = config.get("fluid.temperature")
        solid_temp = config.get("solid.temperature")

        fluid_temp.coupled_to = "solid.temperature"
        solid_temp.coupled_to = "fluid.temperature"
```

---

## Migration Checklist

- [x] Create `configurable.py` with `Configurable` type alias
- [x] Rename `registry.py` → `config_context.py`
- [x] Rename `ModelRegistry` → `ConfigContext`
- [x] Add dot notation parsing to `ConfigContext.get()`
- [x] Update `initializer.py` to use `ConfigContext`
- [x] Update `__init__.py` exports (both framework and initialization)
- [x] Update `incompressibleFluid.py`
- [x] Create `test_config_context.py` with new tests
- [x] Create `test_configurable_field.py` with new tests
- [x] Update `test_fixtures.py`
- [x] Update `test_incompressible_fluid.py`
- [x] Update `test_configure_stage.py` and `test_resolve_dependencies_stage.py`
- [x] Update `model_protocol.py` docstrings
- [x] Update `initialization.rst` documentation
- [x] Run all tests (59/59 passed)
- [ ] Delete deprecated `AdaptableField` function (kept for backward compatibility)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing code | High | Not a concern (per user) |
| Test failures | Medium | Update all tests in same PR |
| Documentation drift | Low | Update docs with code |

---

## Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1: Core Changes | 2 hours |
| Phase 2: Initializer | 1 hour |
| Phase 3: Solvers | 1 hour |
| Phase 4: Tests | 2 hours |
| Phase 5: Documentation | 1 hour |
| **Total** | **7 hours** |
