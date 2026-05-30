<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# Implementation instructions — unified `SolutionLoop`

Step-by-step plan to implement the recommended proposal from
[`python-time-loop.md`](./python-time-loop.md): one `SolutionLoop` (the single
main iteration loop) whose predicate is the control, with `RunTime` / `WriteControl`
/ `TimeStepStrategy` as injected abstractions, supporting steady **and** unsteady
on the NeoN and pybFoam backends.

## Where this lands

The Python package lives on **`stack/python_arch`** (`src/neofoam/…`); the
C++/NeoN tree on `worktree-enh+pytimeloop` does *not* contain it. Branch from
`stack/python_arch`:

```bash
git switch stack/python_arch && git switch -c feat/solution-loop
```

Files touched:

| File | Change |
|------|--------|
| `src/neofoam/framework/runtime.py` *(new)* | `RunTime` protocol |
| `src/neofoam/framework/write_control.py` *(new)* | `WriteControl` + policies + factory |
| `src/neofoam/algorithms/control.py` | add `run()`, `converged()`, `store_residual()`, `time_step_strategy` |
| `src/neofoam/solver/incompressibleFluid/incompressibleFluid.py` | rename `TimeLoop`→`SolutionLoop`; predicate; `set_time_step`; `write_output` |
| `.../models/pressure_velocity/base.py` | re-enable SIMPLE/PISO selection |
| `.../models/pressure_velocity/control_factory.py` | build the right control + strategy + write control |
| `.../models/pressure_velocity/pimpleAlgorithm.py` | publish initial residuals to the control |
| `test/solver/incompressibleFluid/…`, `test/framework/…` | tests |

Work in dependency order (Steps 1→8); each step compiles and keeps `pytest`
green. Build/verify with the project workflow:

```bash
pip install .[all] -v           # rebuild bindings if pybFoam changed
pytest test/ -q
SKIP=reuse pre-commit run --all-files
```

---

## Step 0 — bind `Time::writeAndEnd` in pybFoam (backend prerequisite)

Steady termination needs the backend to end the run. In
`pybFoam/src/pybFoam/pybFoam_core/bind_time.cpp`, add to the `Time` class binding:

```cpp
.def("end", &Foam::Time::writeAndEnd)
```

Rebuild pybFoam (`pip install .` in the pybFoam tree, or `pip install .[all] -v`
in NeoFOAM if it pulls pybFoam). Verify: `python -c "import pybFoam; print(hasattr(pybFoam.Time, 'end'))"`.

---

## Step 1 — `RunTime` protocol (R4: one seam for both backends)

Create `src/neofoam/framework/runtime.py`. Declare **every** accessor the loop,
write control and strategy use (this is the full interface — keep it minimal but
complete):

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunTime(Protocol):
    """Backend-agnostic time object. The SolutionLoop talks only to this."""
    def run(self) -> bool: ...                       # more steps? (pure advance test)
    def increment(self) -> None: ...                 # ++runTime
    def end(self) -> None: ...                       # writeAndEnd (steady termination)
    def write(self, force: bool = False) -> None: ...
    def outputTime(self) -> bool: ...                # backend writeControl decision
    def timeName(self) -> str: ...
    def value(self) -> float: ...                    # current time value
    def timeIndex(self) -> int: ...                  # current step index
    def deltaTValue(self) -> float: ...
    def setDeltaT(self, value: float) -> None: ...
    def printExecutionTime(self) -> None: ...
    @property
    def controlDict(self) -> Any: ...                # the parsed OpenFOAM dict
```

`pybFoam.Time` already satisfies all of these except possibly `timeIndex`
(bind `Time::timeIndex` if missing) — it now has `end` from Step 0. Type
`Context.runtime` as `RunTime` in `src/neofoam/framework/context.py`:

```python
from neofoam.framework.runtime import RunTime
class Context:
    runtime: RunTime
```

`mypy` will now check every `ctx.runtime.*` call against the protocol — fix any
drift it reports rather than widening the protocol.

> The `NeoNRunTime` adapter (wrapping `NeoFOAM::RunTime`) is implemented when the
> NeoN bindings exist; it is a class satisfying this same protocol. Nothing else
> in this plan changes for that backend.

---

## Step 2 — a separate `SolutionControl` owns advance + convergence

In `src/neofoam/algorithms/control.py`, add a **new** `SolutionControl` class —
*not* methods on `PimpleControl`/`SimpleControl`, which stay pure corrector-loop
controls. `SolutionControl` owns the outer-loop predicate; an empty
`residualControl` is transient (never ends early), a populated one is steady
(ends on convergence). It feeds `ResidualConvergenceCondition` from a published
residual store (replacing the `_extract_residual_from_context` `1.0` placeholder):

```python
class SolutionControl(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    residualControl: dict[str, float] = Field(default_factory=dict)

    _residual_check: Optional[ResidualConvergenceCondition] = None
    _residuals: dict[str, float] = {}

    def model_post_init(self, _ctx: Any) -> None:
        self._residuals = {}
        self._residual_check = ResidualConvergenceCondition(residualControl=self.residualControl)
        self._residual_check._get_residual = lambda _c, f: self._residuals.get(f, 1.0)

    def store_residual(self, field: str, initial_residual: float) -> None:
        self._residuals[field] = float(initial_residual)

    def converged(self) -> bool:
        assert self._residual_check is not None
        self._residual_check(None)                 # recompute from stored residuals
        return self._residual_check.converged()    # empty residualControl -> False

    def run(self, runtime: "RunTime") -> bool:
        if self.converged():
            runtime.stop()                          # == runTime.writeAndEnd()
        return bool(runtime.run())
```

`PimpleControl`/`SimpleControl` are **unchanged** (no `run`/`store_residual`) — the
`SolutionLoop` delegates to the `SolutionControl`, which is built alongside the
algorithm control.

> Type note (per repo mypy strict): use the existing
> `model_config = {"arbitrary_types_allowed": True}`; the `RunTime` annotation can
> be a forward-ref string to avoid a framework→algorithms import cycle.

---

## Step 3 — rename `TimeLoop` → `SolutionLoop`; control-driven predicate

In `src/neofoam/solver/incompressibleFluid/incompressibleFluid.py`:

```python
class SolutionLoop:
    """The one main iteration loop (time loop == steady iteration loop).

    The RunTime advances; the control decides termination
    (cf. Foam::pimpleControl::run(Time&) / simpleControl::loop()).
    """
    def __init__(self, control: Any) -> None:
        self._control = control

    def __call__(self, ctx: Context) -> bool:
        return bool(self._control.run(ctx.runtime))
```

Wire it in `execution_graph` — `SolutionLoop` reads the **`SolutionControl`** from
`ctx.models` (built alongside the algorithm control), so there is no build-time
coupling:

```python
def __call__(self, ctx: Context) -> bool:
    return bool(ctx.models["solution_control"].run(ctx.runtime))
```

`SolutionControl` is distinct from `ctx.models["pimple_control"]` (which
`inner_loop` uses for the corrector loop): the outer predicate and the inner
corrector structure are separate concerns.

---

## Step 4 — `WriteControl` abstraction + factory + `write_output`

Create `src/neofoam/framework/write_control.py`:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
from __future__ import annotations
from typing import Protocol, runtime_checkable
from neofoam.framework.runtime import RunTime


@runtime_checkable
class WriteControl(Protocol):
    def should_write(self, runtime: RunTime) -> bool: ...


class BackendWriteControl:
    """Defer to the backend's own writeControl (pybFoam Time.outputTime())."""
    def should_write(self, runtime: RunTime) -> bool:
        return runtime.outputTime()


class IntervalWriteControl:
    def __init__(self, interval: int) -> None:
        self._n = max(1, interval)
    def should_write(self, runtime: RunTime) -> bool:
        return runtime.timeIndex() % self._n == 0


class RunTimeWriteControl:
    def __init__(self, interval: float, start: float) -> None:
        self._dt, self._last = interval, start
    def should_write(self, runtime: RunTime) -> bool:
        if runtime.value() - self._last >= self._dt - 1e-10:
            self._last = runtime.value()
            return True
        return False


def write_control_from_dict(control_dict, start_time: float, *, backend_decides: bool) -> WriteControl:
    if backend_decides:                       # pybFoam Time implements writeControl itself
        return BackendWriteControl()
    kind = str(control_dict.lookupOrDefault("writeControl", "timeStep"))
    interval = control_dict.lookupOrDefault("writeInterval", 1.0)
    if kind == "timeStep":
        return IntervalWriteControl(int(interval))
    return RunTimeWriteControl(float(interval), start_time)
```

Update `write_output` to depend on the abstraction. Today it unconditionally
calls `ctx.runtime.write(True)`; make it ask the write control:

```python
@incompressibleFluid.operation(depends_on=["turbulence_correction"])
def write_output(
    self: Any, ctx: Context, write_control: Annotated[Any, "models"]
) -> None:
    if write_control.should_write(ctx.runtime):
        ctx.runtime.write()
    ctx.runtime.printExecutionTime()
```

Register `write_control` as a model in the algorithm's `build()` init steps
(alongside `pimple_control`), built from `controlDict`. On the pybFoam backend use
`backend_decides=True` (forwards `Time.outputTime()`), preserving exact OpenFOAM
write semantics.

---

## Step 5 — `TimeStepStrategy`; delete the `SIMPLE` branch (R3)

Add strategies + factory in `control.py` (or a new `time_step.py`):

```python
class FixedTimeStep(BaseModel):
    def adjust(self, ctx: Any) -> None:
        return None

class CourantTimeStep(BaseModel):
    maxCo: float
    maxDeltaT: float
    def adjust(self, ctx: Any) -> None:
        co = ctx.max_courant()               # see note
        rt = ctx.runtime
        if co > 1e-10:
            rt.setDeltaT(min(rt.deltaTValue() * min(self.maxCo / co, 1.2), self.maxDeltaT))


def time_step_strategy(control_dict) -> Any:
    if str(control_dict.lookupOrDefault("adjustTimeStep", "no")).lower() in {"yes", "true", "on"}:
        return CourantTimeStep(
            maxCo=float(control_dict.lookup("maxCo")),
            maxDeltaT=float(control_dict.lookupOrDefault("maxDeltaT", 1e30)),
        )
    return FixedTimeStep()
```

Replace `set_time_step` — **remove** the `algorithm_type == "SIMPLE"` branch; the
strategy is mode-agnostic (steady gets `FixedTimeStep`, which is a no-op):

```python
@incompressibleFluid.operation()
def set_time_step(self: Any, ctx: Context, time_step: Annotated[Any, "models"]) -> None:
    time_step.adjust(ctx)
```

Register `time_step` as a model in `build()`. For the Courant number, expose a
backend helper (pybFoam `computeCFLNumber`, NeoN `computeCoNum`) as a small method
on `Context` (`max_courant()`), or read it from `ctx` where the CFL condition
already computes it. Keep the choice consistent with how `cfl_condition` is
provided today.

---

## Step 6 — publish initial residuals (enables steady convergence)

In `pimpleAlgorithm.py`, capture the initial residual from each solve and hand it
to the control via `store_residual`. `pyf.solve(...)` and `pEqn.solve(...)` return
a `SolverPerformance` with `initialResidual()`:

```python
# momentum(): capture U residual
if pimple_control.momentumPredictor():
    perf = pyf.solve(UEqn + fvc.grad(p))
    pimple_control.store_residual("U", float(perf.initialResidual()))

# continuity(): capture p residual on the first non-ortho solve
perf_p = pEqn.solve(p.select(pimple_control.finalInnerIter()))
pimple_control.store_residual("p", float(perf_p.initialResidual()))
```

For PIMPLE these calls are no-ops (Step 2b). For SIMPLE they feed `converged()`,
so `SolutionLoop` ends the run once every `residualControl` field is below
tolerance. (If `SolverPerformance` exposes a vector of component residuals for
`U`, take the max — match `simpleControl::criteriaSatisfied()` behaviour.)

---

## Step 7 — actually select SIMPLE / PISO (unification)

`models/pressure_velocity/base.py` currently detects `algorithm_type` but falls
back to PIMPLE for SIMPLE/PISO. Make the detected type pick the matching control
in `control_factory.py`, and expose it under the **same** `ctx.models` key the
loop/inner-loop read (`pimple_control` — or generalise the key to `control`):

```python
# control_factory.py
def create_control(context: dict[str, Any]) -> Any:
    algo = pimple.algorithm_type            # "PIMPLE" | "SIMPLE" | "PISO"
    if algo == "SIMPLE":
        return create_simple_control(context)   # SimpleControl(residualControl=...)
    # PISO == PIMPLE with nOuterCorrectors == 1
    return create_pimple_control(context)
```

- **PISO** — `create_pimple_control` with `nOuterCorrectors=1` (read from dict;
  PISO dict has none → default 1). The inner outer-corrector loop runs once.
- **SIMPLE** — `SimpleControl` with `residualControl` from the `SIMPLE` subdict;
  `run()` ends the run on convergence; pair with `FixedTimeStep` automatically
  (because `adjustTimeStep` is `no` in steady cases).

Confirm `inner_loop` (`ctx.models["..."].loop()`) and `SolutionLoop`
(`ctx.models["..."].run(ctx.runtime)`) read the same key. Both `PimpleControl` and
`SimpleControl` already implement `loop()`; both now implement `run()`.

No `if steady:` anywhere in the solver — the case's `fvSolution` (`PIMPLE` vs
`SIMPLE`) + `fvSchemes` (`ddtSchemes steadyState`) decide everything.

---

## Step 8 — backend selection (forward-compatible)

Add a `backend` argument threaded from `run(argv, backend="pybfoam")` to `Context`
construction. Today only `pybfoam` exists; the `RunTime` typing (Step 1) means the
NeoN path is a drop-in once `NeoNRunTime` is implemented. Keep the signature now so
callers don't change later:

```python
def run(argv=None, log_file=None, backend: str = "pybfoam") -> Context:
    ...
    solver = incompressibleFluid.instantiate(argv=argv or [], backend=backend)
```

---

## Tests

Add under `test/` (mirror `src/` 1:1; free functions, not classes):

1. **`test/framework/test_write_control.py`** — `IntervalWriteControl` /
   `RunTimeWriteControl` / `BackendWriteControl` against a fake `RunTime`
   (a small dataclass implementing the protocol). Assert exact write steps.
2. **`test/algorithms/test_control.py`** — `SolutionControl`: empty
   `residualControl` advances to `endTime` (transient); populated + stored
   residuals < tol calls `runtime.stop()` (steady); partial convergence keeps
   running; `PimpleControl` has no `run`/`store_residual` (separation).
3. **`test/solver/incompressibleFluid/test_solution_loop.py`** —
   - transient case runs to `endTime` (count iterations == `endTime/deltaT`);
   - steady case (`SIMPLE` + `residualControl`, fake residuals decaying below tol)
     calls `runtime.end()` *before* `endTime` and stops;
   - `set_time_step` is a no-op under `FixedTimeStep` and rescales `deltaT` under
     `CourantTimeStep`.
4. **Integration** — run `tutorials/cavity` (transient) and a steady variant
   (`pitzDaily` with a `SIMPLE` `fvSolution` + `steadyState` ddt) through
   `incompressibleFluid.run(["."])`; assert the steady run's `timeName()` at exit
   is < `endTime`.

Fake backend for unit tests:

```python
@dataclass
class FakeRunTime:
    _t: float = 0.0; dt: float = 0.1; end: float = 1.0
    _idx: int = 0; _ended: bool = False
    def run(self): return (not self._ended) and self._t < self.end - 1e-10
    def increment(self): self._t = min(self._t + self.dt, self.end); self._idx += 1
    def end(self): self._ended = True            # type: ignore[no-redef]
    def value(self): return self._t
    def timeIndex(self): return self._idx
    def deltaTValue(self): return self.dt
    def setDeltaT(self, v): self.dt = v
    def outputTime(self): return False
    def write(self, force=False): ...
    def timeName(self): return f"{self._t:g}"
    def printExecutionTime(self): ...
    @property
    def controlDict(self): return {}
```

## Done-when

- `pytest test/ -q` green; `SKIP=reuse pre-commit run --all-files` clean (format,
  ruff, mypy strict).
- A `PIMPLE` case and a `SIMPLE` case run through the **same** `SolutionLoop` and
  graph; the steady case terminates on residual convergence before `endTime`.
- No `algorithm_type == "SIMPLE"` (or any steady/transient) branch remains in the
  solver operations.
