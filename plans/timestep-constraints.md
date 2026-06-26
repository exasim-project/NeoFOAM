# Specification: time-step constraints & loop conditions (uses the `Interface` mechanism)

Status: draft · Scope: `neofoam.algorithms.solution_loop` + the incompressibleFluid
time-step models. Supersedes the ad-hoc CFL wiring shipped in commit `e9da83d0e`.

**Depends on `plans/interface-spec.md`** (mechanism implemented in
`src/neofoam/framework/interface/`, iterations 1–4). This spec assumes that mechanism
and relies only on its **public API as actually built**:

- `Interface(name)` → an `InterfaceSpec` (factory mirroring `Model(name)`).
- `@iface.combine` registers the single fold function (`Callable[[Iterable[T]], T]`;
  a second registration raises `RuntimeError`).
- `@iface.contribute` registers an operation-style contribution; a contribution that
  declares a `Context`-typed parameter raises `ValueError` at registration.
- **Injection:** a parameter annotated with the **`InterfaceSpec` instance** receives a
  `BoundInterface`; **calling** it returns `iface.collect(ctx)` over the active
  contributions. Requires (a) the consumer module does **not** use `from __future__
  import annotations` (PEP 563 stringifies the annotation and defeats instance
  detection), and (b) the interface is present in `ctx.interfaces` (placed by
  `interface_step`), else resolution raises.
- `iface.collect(ctx)` folds the active contributions, resolving each contribution's
  parameters through the shared `DependencyResolver` (the `@model.operation` path).
- `iface.activate(fn)` / `iface.deactivate(fn)` toggle a contribution's participation;
  all contributions are active by default.

It does not re-specify that mechanism.

## 1. Motivation

The current adaptive-time-step design works but the *control-loop interface* is weak
(reviews `review/adaptive-time-step/iter-1..4`):

- The loop hardcodes `min` aggregation — open to new constraints, **closed to new
  combination logic** (AND/OR/conditional). OCP holds on one axis only.
- The constraint↔measurement link is **name-matched and optional**: `set_time_step`
  fires *every* `measurement_provider.*`, and `DeltaTConstraint.measures` is declared
  but never read, so a missing/misnamed provider **silently disables** the rule
  (`0.0` → `VGREAT` → no limit).
- Adding a rule needs boilerplate: `install_constraints_step`, a
  `measurement_provider.<name>` node, and hand-written `depends_on`.
- A stale `set_courant` docstring reference remains.

Re-express time-step **constraints** and loop **conditions** as contributions to two
`Interface` gather points, consumed by `solutionLoop` via injection — so a rule is one
optional model + one operation-style function, and the loop is open to both new rules
and new combinations.

## 2. Requirements

Each requirement is atomic, has a priority (**MUST** / **SHOULD** / **COULD**), and a
verification (the test that proves it). IDs are stable (`TC<n>`).

### Interfaces (the two gather points this feature defines)

| ID | Priority | Requirement | Verify |
|----|----------|-------------|--------|
| TC1 | MUST | The feature defines `timeStepConstraint = Interface("timeStepConstraint")` whose `@combine` fold returns `min(limits, default=VGREAT)` over `Iterable[float]`; contributions return a `float` (max permitted `deltaT`; `VGREAT` = no opinion). | Fold over no values returns `VGREAT`; fold over `[1.0, 2.0]` returns `1.0`. |
| TC2 | MUST | The feature defines `loopCondition = Interface("loopCondition")` whose `@combine` fold returns `all(flags)` over `Iterable[bool]`; contributions return a `bool`. | Fold over no values returns `True`; fold over `[True, False]` returns `False`. |

### Consumer (`solutionLoop`)

| ID | Priority | Requirement | Verify |
|----|----------|-------------|--------|
| TC3 | MUST | `solutionLoop` remains a `Model`; it consumes both interfaces by typing two loop-body-operation parameters with the spec instances and obtaining values by **calling** them. It does not own them. | With both interfaces injected and no contributions, the operation sets `next_dt = VGREAT` and `running = True`. |
| TC4 | MUST | The `solutionLoop` module declaring the injected operation does **not** use `from __future__ import annotations`. | An injection test in that module resolves the `BoundInterface` (would fail if the annotation were stringified). |
| TC5 | MUST | Both interfaces are placed in `ctx.interfaces` (via `interface_step`) before the loop operation runs. | Injecting an interface absent from `ctx.interfaces` raises at resolution. |

### Contributions (the rules)

| ID | Priority | Requirement | Verify |
|----|----------|-------------|--------|
| TC6 | MUST | A constraint/condition is an optional `Model(...).register_with(<family>)` owning one `@IOStrategy` `BaseConfig`; it is active iff its config is present (no `detect()`/toggle metadata in the decorator). | With the config in the case the model appears in the type-driven catalog; without it, it does not. |
| TC7 | MUST | A contribution is registered with `@timeStepConstraint.contribute` / `@loopCondition.contribute`, is operation-style (its fields, the current `deltaT`, and its config are injected), and **never declares a `Context` parameter**. | A contribution with a `Context`-typed parameter raises `ValueError` at registration (decoration time). |
| TC8 | MUST | If a contribution declares a parameter no provider/Context supplies, **folding the interface raises `ValueError`** naming interface + contribution + parameter — never a silent skip. This is a **fold-time** error (raised when `collect(ctx)` runs each step), not build-time. | A contribution needing an absent field raises `ValueError` on `collect(ctx)`. |
| TC9 | SHOULD | The loop exposes its current step as an injectable `DeltaT = Annotated[float, "loop.deltaT"]` so a contribution takes `deltaT: DeltaT` instead of reading `ctx`. (Mechanics open — see §6.) | A contribution typed `deltaT: DeltaT` receives the loop's current step. |

### Behavior

| ID | Priority | Requirement | Verify |
|----|----------|-------------|--------|
| TC10 | MUST | Next `deltaT` is the value of calling the injected `timeStepConstraint()` (fold `min`, `default=VGREAT`); a run with no active constraint is fixed-step. | No active constraint ⇒ `next_dt == VGREAT` (fixed step). |
| TC11 | MUST | The loop continues while the injected `loopCondition()` is `True` (fold `all`, empty ⇒ `True`). | No active condition ⇒ `running == True`. |
| TC12 | SHOULD | A contribution's participation can be toggled via the interface's `activate`/`deactivate`. (The auto-gating of a contribution from its model's active/config state is **not yet wired** — it depends on finishing IF13 in `interface-spec.md`; until then activation is explicit.) | `deactivate(fn)` excludes `fn` from the fold; `activate(fn)` re-includes it. |
| TC13 | COULD | Logical combinators (`all_` / `any_` / `when`) compose sub-rules **within** one contribution; a new combinator is a new helper with no loop edit. | A composed contribution folds to the expected limit; adding a combinator touches no loop code. |

### Migration / quality

| ID | Priority | Requirement | Verify |
|----|----------|-------------|--------|
| TC14 | MUST | The existing CFL limit ships as a `@timeStepConstraint.contribute`. | CFL contribution active ⇒ `deltaT` limited as before. |
| TC15 | MUST | The existing `maxDeltaT` limit ships as a `@timeStepConstraint.contribute`. | `maxDeltaT` contribution caps `deltaT`. |
| TC16 | MUST | `DeltaTConstraint.measures` is deleted. | Symbol absent (grep / import error). |
| TC17 | MUST | The `measurement_provider.*` nodes are deleted. | No `measurement_provider.` node registered. |
| TC18 | MUST | `install_constraints_step` is deleted. | Symbol absent. |
| TC19 | SHOULD | Constructor DI `SolutionLoop(constraints=[...], conditions=[...])` is retained for tests/embedding. | A loop built via the constructor folds the passed rules. |
| TC20 | MUST | The stale `:meth:\`set_courant\`` docstring reference is corrected. | No `set_courant` reference remains. |
| TC21 | MUST | The framework solution loop stays pure-Python; pybFoam CFL/alpha-CFL contributions live solver-side and inject pybFoam fields. | `algorithms.solution_loop` imports no `pybFoam`; CFL contribution lives under the solver. |

## 3. Consumer wiring

`solutionLoop` injects the two interfaces in its loop-body operation and reads the
folds by calling them; nothing else changes in the engine.

```python
# algorithms/solution_loop/interfaces.py — Interface(name) + @combine (like a ModelSpec)
from typing import Iterable
from neofoam.framework.interface import Interface

timeStepConstraint = Interface("timeStepConstraint")
loopCondition = Interface("loopCondition")

@timeStepConstraint.combine
def fold_min(limits: Iterable[float]) -> float:
    return min(limits, default=VGREAT)   # empty -> VGREAT (fixed step)

@loopCondition.combine
def fold_all(flags: Iterable[bool]) -> bool:
    return all(flags)                    # empty -> True (keeps running)
```

```python
# algorithms/solution_loop/solution_loop.py
# NOTE: this module must NOT use `from __future__ import annotations` — the
# interface parameters below are annotated with InterfaceSpec *instances*, which
# PEP 563 would stringify and the resolver would no longer detect (TC4).
from .interfaces import timeStepConstraint, loopCondition

@solutionLoop.operation
def set_time_step(self, constraints: timeStepConstraint, conditions: loopCondition) -> None:
    self.next_dt = constraints()         # BoundInterface.__call__ -> collect(ctx) -> min
    self.running = conditions()          # -> all conditions True
```

`DeltaT` marker (TC9) so contributions stay `ctx`-free — **open** (§6); final form TBD
once the resolver supports an app-level `Annotated` key:

```python
DeltaT = Annotated[float, "loop.deltaT"]   # injected with the loop's current step
```

## 4. User-facing code

A constraint = one optional model + one config + one operation-style contribution.

```python
# models/courant.py  — CFL constraint (active when CourantConfig is in the case)
@IOStrategy(OF("system/controlDict"))
class CourantConfig(BaseConfig):
    maxCo: float = Field(gt=0)
    maxDeltaT: float | None = Field(default=None, gt=0)

courant = Model("courant").register_with(incompressibleFluidModel)
courant.config(CourantConfig)

@timeStepConstraint.contribute
def courant_limit(phi: surfaceScalarField, deltaT: DeltaT, cfg: CourantConfig) -> float:
    Co = computeCFLNumber(phi)[0]
    return deltaT * cfg.maxCo / Co if Co > SMALL else VGREAT
```

```python
# models/interface_courant.py — a second rule, zero core-loop edits
@IOStrategy(OF("system/controlDict"))
class AlphaCourantConfig(BaseConfig):
    maxAlphaCo: float = Field(gt=0)

alphaCo = Model("interfaceCourant").register_with(incompressibleFluidModel)
alphaCo.config(AlphaCourantConfig)

@timeStepConstraint.contribute            # declares alphaPhi → ValueError at collect if absent (TC8)
def alpha_courant(alphaPhi: surfaceScalarField, deltaT: DeltaT, cfg: AlphaCourantConfig) -> float:
    Co = computeAlphaCFLNumber(alphaPhi)[0]
    return deltaT * cfg.maxAlphaCo / Co if Co > SMALL else VGREAT
```

```python
# models/divergence_guard.py — a loop CONDITION (stops the run)
@IOStrategy(OF("system/controlDict"))
class DivergenceConfig(BaseConfig):
    maxCo: float = Field(default=1e3, gt=0)

guard = Model("divergenceGuard").register_with(incompressibleFluidModel)
guard.config(DivergenceConfig)

@loopCondition.contribute                 # loop runs while this is True
def not_diverged(phi: surfaceScalarField, cfg: DivergenceConfig) -> bool:
    return computeCFLNumber(phi)[0] < cfg.maxCo
```

Optional composition within one contribution:

```python
@timeStepConstraint.contribute
def adaptive(phi: surfaceScalarField, deltaT: DeltaT, cfg: CourantConfig) -> float:
    return all_(courant_limit, when(have("alphaPhi"), alpha_courant), max_delta_t)
```

Tests / embedding still use plain DI:

```python
loop = SolutionLoop(state, constraints=[courant_limit], conditions=[not_diverged])
```

## 5. Acceptance criteria

Each maps to requirements above and is a test, not a sentiment.

- **AC1 (TC6, TC7)** A constraint/condition is added by writing **one config + one
  `@<interface>.contribute` function** — no `depends_on`, no provider node, no
  `install_constraints_step`.
- **AC2 (TC7, TC8)** The contribution is operation-style (injected fields / `deltaT` /
  config; never `ctx`); a missing declared field raises `ValueError` **when the
  interface is folded** (`collect(ctx)`), naming the interface, contribution, and
  parameter.
- **AC3 (TC3, TC10, TC11)** `solutionLoop` injects `timeStepConstraint` / `loopCondition`
  (typed with the spec instances) and **calls** them; with no active constraint the run
  is fixed-step (`min(..., default=VGREAT)`), with no condition it runs (`all([]) == True`).
- **AC4 (TC6)** Selecting the model in the wizard / writing the config activates the
  rule; omitting it leaves a fixed-step run.
- **AC5 (TC13)** `all_` / `any_` / `when` compose within a contribution; `min` is the
  default fold.
- **AC6 (TC14–TC18, TC20)** CFL + `maxDeltaT` ship as contributions; `measures`,
  `measurement_provider.*`, `install_constraints_step` are deleted; the `set_courant`
  docstring is fixed.

## 6. Out of scope / open questions

- **`DeltaT` marker mechanics (TC9).** The `Interface`/`DependencyResolver` as built
  resolves `Depends(...)`, `Context`, and `Annotated[..., "fields"/"models"]`; an
  app-level key such as `Annotated[float, "loop.deltaT"]` is **not yet supported** — the
  loop's mechanism for publishing its current step as an injectable is open.
- **Auto-gating contributions (TC12).** `activate`/`deactivate` exist, but wiring a
  contribution's active state to its model's config-presence/`@detect` is the unfinished
  IF13 work in `interface-spec.md`; until then activation is explicit.
- **Field/config injection parity** (`Annotated[..., "fields"]` for a real
  `surfaceScalarField`, config objects) rides on the IF8/IF11 parity work in
  `interface-spec.md`.
- **Per-step dataflow DAG** is deferred (start with fold-time resolution + calling the
  injected interface).
- **Family placement:** keep contributions in `incompressibleFluidModel` (solver-local),
  or promote to a framework-level time-step family for cross-solver reuse.
- Anything about the `Interface` spec itself (its API, injection-key typing, Context
  placement) is owned by `plans/interface-spec.md`.
