# Initialization Framework: Architecture & Code Quality Proposals

## Summary (Top Table)

| ID | Area | Current Risk / Smell | Proposal | Expected Benefit | Effort | Priority |
|---|---|---|---|---|---|---|
| P1 | API design | Two competing styles (decorator-based `StagedInit` and explicit helper flow) increase cognitive load | Define one canonical public API and remove the legacy style | Lower onboarding cost, fewer integration mistakes | M | High |
| P2 | Type safety | Heavy `Any` usage and untyped string keys (`"fields.U"`, `"models.x"`) | Introduce typed aliases / enums and stricter signatures for step names and categories | Earlier error detection, better IDE support | M | High |
| P3 | Dependency graph | Runtime validation spread across functions; errors are good but not centralized | Add a dedicated `InitGraphValidator` with structured diagnostics | Consistent validation and easier debugging/reporting | M | High |
| P4 | `InitStep` execution model | Arity introspection (`inspect.signature`) is fragile and runtime-only | Replace with explicit initializer protocol (`NoArgInit` / `ContextInit`) | Simpler semantics and fewer edge cases | M | High |
| P5 | Context routing | Prefix-based routing in `build_context_from_objects` is stringly-typed | Use explicit step `category` only (no prefix fallback) | More robust routing and safer refactors | S | High |
| P6 | `ConfigContext` semantics | Region/path disambiguation is implicit; missing duplicate-policy controls | Add strict mode (`get_or_raise`, duplicate policy, explicit path object) | Prevent silent misconfiguration | M | Medium |
| P7 | Builder behavior | `_normalize_lazy_init` auto-prefixes unknown names to `fields.` (surprising default) | Make normalization explicit and configurable (`strict=True`) | Reduced hidden behavior and clearer intent | S | Medium |
| P8 | Dependency object | `Depends.scope` accepts free strings without validation | Validate scope via `Literal`/Enum and validate at construction | Eliminates invalid scopes early | S | Medium |
| P9 | State mutation | `StagedInit.run()` mutates solver config with hard-coded keys | Move config export to dedicated adapter (`ContextExportPolicy`) | Cleaner separation of concerns | M | Medium |
| P10 | Testing strategy | Good unit coverage, but little contract/property testing | Add property-based tests and API contract tests for compatibility paths | Better regression resistance | M | Medium |
| P11 | Observability | Minimal graph/state diagnostics beyond exceptions | Add trace logging + optional graph dump (DOT/JSON) | Faster incident diagnosis | S | Medium |
| P12 | Dependency footprint | `networkx` used for relatively small DAG operation | Keep `networkx` for now; evaluate lightweight internal topo sort behind interface | Lower dependency risk while preserving behavior | M | Low |

---

## Scope

These proposals target:
- `neofoam.framework.initialization`
- related tests in `test/initialization`

Goal: make initialization easier to reason about, safer to extend, and simpler to test.

Assumption for this document: a clean break is acceptable; backward compatibility is not required.

---

## Detailed Proposals

### P1 — Converge to one canonical initialization API

**Observation**
- The package documents a hybrid explicit approach but still exposes decorator-driven staged initialization.
- Both are valid, but supporting both equally can fragment usage patterns.

**Proposal**
1. Pick one **canonical** style for new code (recommended: explicit staged functions + builder).
2. Remove the non-canonical style from the public API.
3. Add one architecture decision record (ADR) documenting why.

**Acceptance criteria**
- README + docs show one recommended pattern.
- Legacy API entry points are removed.

**Source snippet (current)**

```python
# decorator style (to be removed)
init = StagedInit("MySolver")

@init.load
def load() -> LoadResult:
   ...

@init.build
def build(core, opt) -> list[InitStep]:
   ...
```

**Source snippet (proposal)**

```python
# canonical explicit style
def load_solver() -> LoadResult:
   ...

def resolve_solver(config: ConfigContext, load_result: LoadResult) -> None:
   ...

def build_solver(load_result: LoadResult) -> list[InitStep]:
   ...

ctx = execute_initialization(build_solver(load_solver()))
```

---

### P2 — Strengthen type contracts and remove stringly-typed hotspots

**Observation**
- Many signatures use `Any` and plain strings for path/category semantics.

**Proposal**
1. Introduce type aliases:
   - `InitName = NewType("InitName", str)`
   - `DependencyPath = NewType("DependencyPath", str)`
2. Add `Literal` or `Enum` for category and dependency scope.
3. Replace broad `list` annotations with precise types in `StagedInit` and `LoadResult`.

**Acceptance criteria**
- Static checker flags invalid category/scope values.
- Public API signatures are fully typed.

**Source snippet (current)**

```python
def register_step(name: str, category: str, depends_on: list[str]) -> None:
   ...
```

**Source snippet (proposal)**

```python
from typing import NewType, Literal

InitName = NewType("InitName", str)
DependencyPath = NewType("DependencyPath", str)
InitCategory = Literal["fields", "models", "operators", "resource"]

def register_step(name: InitName, category: InitCategory, depends_on: list[DependencyPath]) -> None:
   ...
```

---

### P3 — Centralize graph validation

**Observation**
- Validation logic exists, but diagnostics are spread between sort and helper validation.

**Proposal**
Create `InitGraphValidator` that returns a structured report:
- duplicates
- missing deps
- cycles
- optional warnings (unreachable nodes, naming collisions)

**Acceptance criteria**
- `execute_initialization()` validates through a single entry point.
- Errors preserve machine-readable structure for tests and tooling.

**Source snippet (current)**

```python
errors = validate_lazy_init_graph(lazy_inits)
if errors:
   raise ValueError(errors)
```

**Source snippet (proposal)**

```python
report = InitGraphValidator.validate(lazy_inits)
if not report.is_valid:
   raise InitializationGraphError(report)
```

---

### P4 — Replace arity-based dispatch with explicit callable protocols

**Observation**
- `_dispatch_by_arity()` and `InitStep.__post_init__()` infer behavior by parameter count.

**Proposal**
Define explicit protocols/wrappers:
- `NoArgInitializer`
- `ContextInitializer`

Or store `requires_context: bool` explicitly in `InitStep`.

**Acceptance criteria**
- No runtime behavior branch based only on parameter count.
- Clear error message when wrong initializer type is provided.

**Source snippet (current)**

```python
step = InitStep(name="fields.U", depends_on=["mesh"], initializer=lambda ctx: make_u(ctx["mesh"]))
value = step.execute(context={"mesh": mesh})
```

**Source snippet (proposal)**

```python
from typing import Literal

InitMode = Literal["no_context", "context"]

step = InitStep(
   name="fields.U",
   depends_on=["mesh"],
   initializer=lambda ctx: make_u(ctx["mesh"]),
   mode="context",
)
value = step.execute(context={"mesh": mesh})
```

---

### P5 — Prefer `category` over name-prefix routing

**Observation**
- `build_context_from_objects()` routes by string prefixes.

**Proposal**
1. Route using `InitStep.category` as source of truth.
2. Keep prefix parsing only as backward-compatible fallback.
3. Add warning when fallback is used.

**Acceptance criteria**
- Category-driven routes are covered by tests.
- Prefix-only names are rejected with a clear error.

**Source snippet (current)**

```python
objects = {"fields.U": U, "operators.momentum": mom}
ctx = build_context_from_objects(objects)  # routed by name prefix
```

**Source snippet (proposal)**

```python
items = [
   BuiltObject(name="U", category="fields", value=U),
   BuiltObject(name="momentum", category="operators", value=mom),
]
ctx = build_context_from_built_objects(items)  # routed by explicit category
```

---

### P6 — Make `ConfigContext` safer in strict mode

**Observation**
- `get()` returns `None` silently; path disambiguation is clever but implicit.

**Proposal**
Add strict APIs and policy:
- `get_or_raise(path)`
- `register(..., on_duplicate={"error"|"replace"|"ignore"})`
- optional explicit `ConfigPath(region, name)` object

**Acceptance criteria**
- Strict mode is optional but documented.
- Duplicate handling and missing-model behavior are deterministic.

**Source snippet (current)**

```python
config.register("transport", t1)
config.register("transport", t2)      # overwrite
missing = config.get("not_there")      # None
```

**Source snippet (proposal)**

```python
config.register("transport", t1, on_duplicate="error")
config.register("transport", t2, on_duplicate="error")  # raises ValueError

transport = config.get_or_raise("transport")
missing = config.get_or_raise("not_there")               # raises KeyError
```

---

### P7 — Remove surprising normalization defaults in builder

**Observation**
- `_normalize_lazy_init()` forces unknown names to `fields.`.

**Proposal**
1. Make normalization strategy explicit on `InitializerBuilder` construction.
2. Default to `strict=True` for new call sites.
3. If unknown category/name pattern appears, raise with migration hint.

**Acceptance criteria**
- No hidden recategorization in strict mode.
- Non-strict normalization path is removed.

**Source snippet (current)**

```python
li = InitStep("U", initializer=lambda: value)
normalized = builder._normalize_lazy_init(li)
assert normalized.name == "fields.U"  # implicit rewrite
```

**Source snippet (proposal)**

```python
builder = InitializerBuilder(strict=True)
li = InitStep("U", initializer=lambda: value)

builder.normalize(li)  # raises ValueError: unknown unqualified name in strict mode
```

---

### P8 — Validate `Depends` metadata

**Observation**
- `Depends.scope` is unconstrained string.

**Proposal**
Use:
- `Literal["time_step", "iteration", "operation"]`
- constructor validation for unsupported scopes

**Acceptance criteria**
- Invalid scopes fail at construction time.
- Unit tests cover invalid values.

**Source snippet (current)**

```python
dep = Depends("fields.U", scope="typo_scope")  # accepted
```

**Source snippet (proposal)**

```python
from typing import Literal

Scope = Literal["time_step", "iteration", "operation"]
dep = Depends("fields.U", scope="time_step")
bad = Depends("fields.U", scope="typo_scope")  # type/runtime error
```

---

### P9 — Isolate side effects in `StagedInit.run()`

**Observation**
- `run()` mutates `self.configs` based on hard-coded model keys (`config`, `solver_config`).

**Proposal**
Move this to pluggable export policy:
- `ContextExportPolicy.export(ctx, solver_state)`

**Acceptance criteria**
- `run()` orchestrates only stage execution.
- Config export logic is independently testable.

**Source snippet (current)**

```python
ctx = execute_initialization(lazy_inits)
if "config" in ctx.models:
   self.configs["solver"] = ctx.models["config"]
```

**Source snippet (proposal)**

```python
ctx = execute_initialization(lazy_inits)
ContextExportPolicy().export(ctx, self.state)
```

---

### P10 — Expand tests from unit to contract/property level

**Observation**
- Existing unit tests are solid and readable.

**Proposal**
Add:
1. Property-based DAG tests (random acyclic graphs, deterministic ordering invariants).
2. API conformance tests for the new canonical surface only.
3. Mutation-style tests around duplicate/missing dependency handling.

**Acceptance criteria**
- At least one property-based suite for graph behavior.
- CI validates only the new canonical API.

**Source snippet (current)**

```python
def test_cycle_raises():
   with pytest.raises(ValueError):
      topological_sort(inits)
```

**Source snippet (proposal)**

```python
from hypothesis import given

@given(acyclic_init_graphs())
def test_topological_order_respects_dependencies(graph):
   order = topological_sort(graph.steps)
   assert is_valid_topological_order(order, graph.edges)
```

---

### P11 — Improve observability and diagnostics

**Proposal**
Add optional debug output:
- stage timing (`load`, `resolve`, `build`, `execute`)
- graph summary (#nodes, #edges, roots, leaves)
- optional graph export (DOT/JSON)

**Acceptance criteria**
- Debug mode can be enabled without changing solver code.
- Failures include graph context snippets.

**Source snippet (current)**

```python
ctx = execute_initialization(lazy_inits)
```

**Source snippet (proposal)**

```python
with init_tracing(enabled=True) as trace:
   ctx = execute_initialization(lazy_inits)

logger.info("init metrics: %s", trace.summary())
trace.dump_graph("init_graph.json")
```

---

### P12 — Reduce coupling to graph implementation

**Proposal**
Introduce interface boundary:
- `TopologicalSorter` protocol with `sort(steps)`
- default implementation uses `networkx`
- optional internal lightweight implementation can be benchmarked later

**Acceptance criteria**
- Sorting backend is swappable without changing callers.
- Behavior and error semantics remain stable.

**Source snippet (current)**

```python
graph = nx.DiGraph()
...
sorted_names = list(nx.lexicographical_topological_sort(graph))
```

**Source snippet (proposal)**

```python
class TopologicalSorter(Protocol):
   def sort(self, steps: list[InitStep]) -> list[InitStep]: ...


sorter: TopologicalSorter = NetworkxTopologicalSorter()
ordered_steps = sorter.sort(lazy_inits)
```

---

## Suggested rollout plan

### Phase 1 (quick wins, 1–2 PRs)
- P8, P11, partial P2 (typing aliases)
- Documentation update defining canonical API and removals (P1)

### Phase 2 (core hardening, 2–4 PRs)
- P3, P4, P5, P6
- Add canonical API conformance tests (P10)

### Phase 3 (cleanup and deprecation)
- P7, P9, P12
- remove remaining legacy entry points and code paths (P1)

---

## Notes for maintainers

- Apply changes directly to the target architecture (clean break).
- Favor simplification over transitional shims.
- Keep error messages stable where tests already assert on message fragments.
