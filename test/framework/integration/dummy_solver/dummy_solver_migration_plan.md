# Dummy Solver Migration Plan

## Summary

The `dummy_solver` integration package is currently blocked by framework API drift (imports, staged-init internals, and removed `LazyInit` path). In addition, graph handling has been refactored into the `neofoam.framework.graph` package, so old `DAGResolver`-based assumptions must be migrated to the new graph utilities.

## Current status (live)

| Work item | Status | Notes |
|---|---|---|
| Move graph resolver code to `framework/graph` | ✅ Done | `DAGResolver` implementation moved to `src/neofoam/framework/graph/dag_resolver.py`; imports updated in solver + dummy solver code |
| Remove duplicated graph functionality from `operations.py` | ✅ Done | Graph-specific resolver logic removed from `src/neofoam/framework/operations.py` |
| BaseConfig import drift | ✅ Done | `src/neofoam/framework/model_factory.py` now imports `BaseConfig` from `neofoam.io` |
| Restore missing dependency resolver module | ✅ Done | Added `src/neofoam/framework/dependency_resolver.py` |
| Migrate `LazyInit` usage to `InitStep` | 🟡 In progress | Core files migrated; pending full runtime verification |
| StagedInit hook API migration in tests | ✅ Done | `test_staged_init.py` now checks `init._hooks.*` |
| Wire `LoadResult.validate()` | ✅ Done | `staged_init.LoadResult.validate()` now delegates to `validate_models` |
| Align dummy YAML decorators with current IO API | 🟡 In progress | Removed unsupported YAML kwargs in `dummy_init.py` and `models/model1.py`; re-test pending |
| Dummy solver integration tests | 🔴 Blocked (active) | Current blocker shifted from graph/import errors to remaining runtime/API mismatches; next test run will refine |

## Required changes (top matrix)

| Area | Required change | Files to update | Priority |
|---|---|---|---|
| Graph refactor migration | Replace legacy `DAGResolver` usage with refactored graph flow (`graph.operations_dag.compute_steps_order` and related graph helpers), or add a temporary compatibility adapter mapped to new graph modules | `test/framework/integration/dummy_solver/dummy_solver.py`, `test/framework/integration/dummy_solver/test_dummy_solver.py`, `src/neofoam/framework/graph/operations_dag.py`, optionally compatibility layer in `src/neofoam/framework/operations.py` | P0 |
| Removed lazy init module | Replace `neofoam.framework.initialization.lazy_init.LazyInit` usage with `InitStep`/helpers (`field`, `model`, `lazy`) | `test/framework/integration/dummy_solver/dummy_init.py`, `test/framework/integration/dummy_solver/models/model1.py`, `test/framework/integration/dummy_solver/models/model2.py` | P0 |
| BaseConfig import drift | Import `BaseConfig` from `neofoam.io` (or `neofoam.io.base`), not `neofoam.io.strategies` | `src/neofoam/framework/model_factory.py` | P0 |
| StagedInit internal API drift | Update tests expecting `_load_func/_resolve_func/_build_func` to `_hooks.load/_hooks.resolve/_hooks.build` | `test/framework/integration/dummy_solver/test_staged_init.py` | P0 |
| Validation execution | Implement or wire `LoadResult.validate()` to use `neofoam.io.validate_models` | `src/neofoam/framework/initialization/staged_init.py` (or adapt tests in `test_validation.py`) | P1 |
| Optional model integration | Ensure `run_build()` outputs are fully compatible with `execute_initialization` and naming conventions | `test/framework/integration/dummy_solver/dummy_init.py`, model files | P1 |
| Final regression check | Re-run integration tests and keep only assertion-level differences | `test/framework/integration/dummy_solver/*` | P1 |

## Old architecture cleanup (remove/combine candidates)

| Candidate | Action | Why | Safety note |
|---|---|---|---|
| `neofoam.framework.initialization.lazy_init` references in dummy solver files | Remove references entirely | Module path no longer exists in current architecture | Safe after `InitStep` migration |
| `LazyInit` type annotations in dummy model/build code | Combine into unified `InitStep`-based return style (`list[InitStep]`) | Avoids dual-init abstractions and matches current framework | Update tests expecting `LazyInit` names/types |
| Direct `StagedInit` private attributes (`_load_func`, `_resolve_func`, `_build_func`) in tests | Remove old assertions and combine on `init._hooks` assertions | Private layout changed; hook container is canonical now | Safe in tests only |
| Legacy `DAGResolver` assumptions in dummy solver | Remove/replace with graph-refactor-native ordering path | Graph ordering logic moved to `framework/graph` package; old API no longer canonical | Keep compatibility shim only short-term if needed |
| Mixed config import styles (`neofoam.io.strategies.BaseConfig` vs `neofoam.io.BaseConfig`) | Combine to single canonical import (`neofoam.io.BaseConfig`) | Reduces import fragility across refactors | Framework-wide consistency preferred |

## Graph refactor notes (new architecture)

- Graph/order responsibilities are now split into dedicated modules under `src/neofoam/framework/graph/`:
  - `operations_dag.py` for DAG build + ordered operation computation
  - `builder.py` and `validator.py` for dependency graph creation/validation
  - `sorter.py` for deterministic topological sorting
- Migration preference for `dummy_solver`:
  1. Build/collect operations as before (`OperationCollection`, `StepBuilder`).
  2. Use graph package ordering utilities for deterministic execution order.
  3. Keep tests asserting order/dependencies, but avoid hard dependency on a removed class name (`DAGResolver`).
- If short-term compatibility is required, implement a thin adapter that delegates to `graph.operations_dag` and mark it for removal after test migration.

## Goal

Get `test/framework/integration/dummy_solver` working against the current `NeoFOAM` framework implementation.

## Current status (verified)

Running:

```bash
uv run pytest test/framework/integration/dummy_solver -q
```

fails during test collection with 4 import/runtime-API breakages:

1. `ImportError: cannot import name 'DAGResolver' from neofoam.framework.operations`
2. `ModuleNotFoundError: No module named neofoam.framework.initialization.lazy_init`
3. `ImportError: cannot import name 'BaseConfig' from neofoam.io.strategies`
4. `StagedInit` internals in tests expect old attributes (`_load_func`, `_resolve_func`, `_build_func`) but current API uses `init._hooks.{load,resolve,build}`

---

## Required changes

### 1) Fix removed/moved imports

- **DAG resolver import path**
  - Replace imports in dummy solver tests/code:
    - from `neofoam.framework.operations import DAGResolver`
    - to `neofoam.framework.graph.dag_resolver import DAGResolver`

- **LazyInit type**
  - `neofoam.framework.initialization.lazy_init.LazyInit` no longer exists.
  - Replace all `LazyInit` usage with `InitStep` helpers:
    - `field(...)`, `model(...)`, `lazy(...)` from `neofoam.framework.initialization`
    - or direct `InitStep(...)` if needed.

- **BaseConfig import source**
  - In framework model factory, import from `neofoam.io` (or `neofoam.io.base`) instead of `neofoam.io.strategies`.
  - This is framework-level and blocks model registration tests.

### 2) Align `dummy_init.py` with current staged-init contract

- Keep using `StagedInit`, but update to current behavior:
  - `init.load` returns `LoadResult(core_models, optional_models)`.
  - `init.resolve` should accept current signature (`config: ConfigContext`) or keep compatibility wrapper.
  - `init.build` must return `list[InitStep]`.
- Replace builder calls that assume `LazyInit` with `InitializerBuilder` + `InitStep` helpers only.

### 3) Update integration tests to current `StagedInit` API

- In `test_staged_init.py`:
  - replace direct checks on:
    - `init._load_func`, `init._resolve_func`, `init._build_func`
  - with checks on:
    - `init._hooks.load`, `init._hooks.resolve`, `init._hooks.build`

### 4) Make validation path executable

- `LoadResult.validate()` is currently a stub raising `NotImplementedError`.
- `test_validation.py` calls `run_load().validate()` and therefore cannot pass until one of these is done:
  1. **Preferred**: implement `LoadResult.validate()` using `neofoam.io.validate_models`.
  2. Alternative: adjust test to validate loaded configs directly via `validate_models(...)`.

### 5) Verify operation execution integration

- Confirm `OperationCollection`/`StepBuilder` + `DAGResolver.resolve(...)` contract still matches expectations in dummy solver tests.
- Keep operation ordering assertions based on `operation_number` and dependency edges.

---

## Execution plan (phased)

## Phase 1 — Unblock test collection

1. Update import paths and graph API usage (`DAGResolver` migration, `BaseConfig`, `LazyInit` references).
2. Replace `LazyInit` return typing and construction in dummy model/build code.
3. Update staged-init attribute assertions in tests.

**Exit criterion:**
`pytest test/framework/integration/dummy_solver -q` reaches test execution (no collection errors).

## Phase 2 — Restore functional behavior

1. Make `dummy_init.py` build stage produce valid `InitStep`s.
2. Ensure optional model `run_build()` outputs are compatible with `execute_initialization`.
3. Verify dependency injection in solver/model operations still resolves field/config parameters.

**Exit criterion:**
Initialization and execution tests run; remaining failures are assertion-level, not API errors.

## Phase 3 — Validation support

1. Implement or wire `LoadResult.validate()`.
2. Ensure invalid config fixtures return deterministic validation errors used by tests.

**Exit criterion:**
`test_validation.py` passes for both valid and invalid config sets.

## Phase 4 — Final stabilization

1. Run full dummy solver integration test folder.
2. Optionally run adjacent initialization unit tests as regression check.
3. Remove temporary compatibility shims if introduced.

**Exit criterion:**
All tests under `test/framework/integration/dummy_solver` pass.

---

## Suggested implementation order (minimal-risk)

1. Framework import fix for `BaseConfig` in model factory.
2. Dummy solver graph migration to refactored `framework/graph` utilities (or temporary adapter).
3. `LazyInit` to `InitStep` migration in dummy files.
4. `StagedInit` test updates.
5. `LoadResult.validate()` wiring.
6. Full test run and cleanup.

---

## Risk notes

- `LoadResult.validate()` touches IO/validation boundaries and may impact existing initialization tests if behavior changes.
- Replacing `LazyInit` with `InitStep` may alter naming conventions (`fields.*`, `models.*`) if not kept consistent.
- Solver/model dependency injection wrappers are sensitive to function signatures; keep test coverage for operation calls with context injection.

---

## Definition of done

- `uv run pytest test/framework/integration/dummy_solver -q` passes.
- No import-time errors in dummy solver integration package.
- Validation tests verify real field-level error metadata for invalid YAML configs.
- Dummy solver run path initializes context and executes resolved operation graph.