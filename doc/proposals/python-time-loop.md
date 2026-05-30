<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# Design proposal: abstracting the OpenFOAM time loop in the Python solver

## Architecture

The design splits the run into five collaborators, each with one reason to
change. The **time loop** advances time and asks two questions — *should I keep
running?* and *should I write now?* — and delegates both to abstractions it does
not own. It never knows the backend, the algorithm mode, or the write policy.

```
                         ┌──────────────────────────────────────────┐
                         │ SolutionLoop  (IterativeOp predicate)          │
                         │  advance time + delegate the two questions │
                         └───────┬───────────────┬──────────────┬─────┘
            "keep running?"      │               │ "write now?" │ "advance/IO"
                                 ▼               ▼              ▼
                     ┌────────────────┐  ┌───────────────┐  ┌──────────────────┐
                     │ Control        │  │ WriteControl  │  │ RunTime (Protocol)│
                     │ (convergence)  │  │ (when to write)│ │ (primitive ops)   │
                     ├────────────────┤  ├───────────────┤  ├──────────────────┤
                     │ PimpleControl  │  │ Interval      │  │ pybFoam Time      │
                     │ SimpleControl  │  │ RunTimeWrite  │  │ NeoNRunTime       │
                     └────────────────┘  │ BackendWrite  │  └──────────────────┘
                                         └───────────────┘
                     ┌──────────────────────────────────────────────┐
                     │ TimeStepStrategy  (Fixed | Courant) — deltaT  │
                     └──────────────────────────────────────────────┘
```

### Time-loop design

`SolutionLoop` is the outer-loop predicate. Its sole responsibility is to *advance
time*; the two decisions it surfaces are pushed onto abstractions:

`SolutionLoop` *is* the single main iteration loop — there is no second loop for
steady state. Its predicate delegates to the control's `run()`, exactly mirroring
`Foam::pimpleControl::run(Time&)` / `simpleControl::loop()`: the control decides
*termination* (reach `endTime` for transient, residual convergence for steady),
the `RunTime` decides *advance*. The loop owns neither policy (DIP) and contains
no steady/transient branch (R3).

```python
class SolutionLoop:
    """The one outer iteration loop. Advances time; owns NO policy.

    Depends on the Control and RunTime *abstractions* (DIP), never on a backend
    or an algorithm mode. Whether each iteration is a physical time step
    (transient) or a relaxation sweep (steady) is the control's concern, not the
    loop's. The body that writes lives in the WriteControl-driven `write_output`
    operation, not here.
    """
    def __init__(self, control: Control) -> None:
        self._control = control                 # injected (DIP)

    def __call__(self, ctx: Context) -> bool:
        return bool(self._control.run(ctx.runtime))   # advance + (steady) converge
```

where the two controls implement `run()` as OpenFOAM does — transient just
advances, steady ends the run on convergence:

```python
class PimpleControl(BaseModel):
    def run(self, runtime: RunTime) -> bool:
        return runtime.run()                # transient: advance to endTime

class SimpleControl(BaseModel):
    def run(self, runtime: RunTime) -> bool:
        if self.converged():                # residuals < residualControl
            runtime.end()                   # == runTime.writeAndEnd()
        return runtime.run()                # otherwise advance (iteration++)
```

### Write-control design

*When* output is persisted is its own responsibility — not the time loop's, not
the solver's. `WriteControl` is a narrow abstraction with concrete policies that
map 1:1 to OpenFOAM's `writeControl` keyword. New policies (e.g. `clockTime`,
`cpuTime`) are added as new classes — the loop is untouched (OCP).

```python
# src/neofoam/framework/write_control.py
from __future__ import annotations
from typing import Protocol, runtime_checkable
from .runtime import RunTime


@runtime_checkable
class WriteControl(Protocol):
    """Single responsibility: decide WHEN to persist fields."""
    def should_write(self, runtime: RunTime) -> bool: ...


class IntervalWriteControl:
    """writeControl timeStep — write every N steps."""
    def __init__(self, interval: int) -> None: self._n = interval
    def should_write(self, runtime: RunTime) -> bool:
        return runtime.timeIndex() % self._n == 0


class RunTimeWriteControl:
    """writeControl runTime/adjustable — write every Δt of simulated time."""
    def __init__(self, interval: float, start: float) -> None:
        self._dt, self._last = interval, start
    def should_write(self, runtime: RunTime) -> bool:
        if runtime.value() - self._last >= self._dt - 1e-10:
            self._last = runtime.value()
            return True
        return False


class BackendWriteControl:
    """Defer to the backend's own writeControl (pybFoam Time.outputTime())."""
    def should_write(self, runtime: RunTime) -> bool:
        return runtime.outputTime()


def write_control_from_dict(control_dict: dict, start_time: float) -> WriteControl:
    kind = str(control_dict.get("writeControl", "timeStep"))
    if kind == "timeStep":
        return IntervalWriteControl(int(control_dict.get("writeInterval", 1)))
    return RunTimeWriteControl(float(control_dict.get("writeInterval", 1.0)), start_time)
```

The `write_output` operation depends on the abstraction, never on a concrete
policy or backend (DIP):

```python
@incompressibleFluid.operation(depends_on=["turbulence_correction"])
def write_output(self, ctx, write_control: Annotated[WriteControl, "models"]) -> None:
    if write_control.should_write(ctx.runtime):   # WHEN — delegated
        ctx.runtime.write()                        # HOW  — backend
    ctx.runtime.printExecutionTime()
```

### Compatibility with the OpenFOAM `controlDict`

The design consumes a **standard, unmodified OpenFOAM `controlDict`** — every
abstraction is constructed from the canonical keys, so an existing case runs
without edits (R1: the `controlDict` is the single source of truth). The `RunTime`
backend reads the dictionary it already owns (`ctx.runtime`), and the policies are
built by factories from the same dictionary:

| `controlDict` key | Consumed by | Effect |
|-------------------|-------------|--------|
| `startTime`, `endTime` | `RunTime.run()` | loop start / stop bound (the iteration cap in steady mode) |
| `deltaT` | `RunTime`, `TimeStepStrategy` | initial step size |
| `writeControl` | `write_control_from_dict` | selects `IntervalWriteControl` (`timeStep`) vs `RunTimeWriteControl` (`runTime`/`adjustable`); pybFoam may use `BackendWriteControl` (`outputTime()`) |
| `writeInterval` | `WriteControl` | every N steps / every Δt of sim time |
| `adjustTimeStep` | `time_step_strategy` | `CourantTimeStep` if `yes`, else `FixedTimeStep` |
| `maxCo`, `maxDeltaT` | `CourantTimeStep` | Courant-limited `deltaT` clamp |
| `runTimeModifiable` | `RunTime` | re-read the dict between steps (backend feature) |
| `writeFormat`, `writePrecision`, `purgeWrite`, `timeFormat`, … | `RunTime.write()` | handled by the backend's IO, unchanged |

```python
# everything is built from the case's controlDict — no hand-written parameters
control_dict = ctx.runtime.controlDict          # the OpenFOAM dictionary, as-is
write_control = write_control_from_dict(control_dict, control_dict["startTime"])
time_step     = time_step_strategy(control_dict)
```

On the **pybFoam** backend the controlDict is parsed by `Foam::Time` itself, so
`writeControl`/`writeInterval` semantics are bit-for-bit OpenFOAM (and
`BackendWriteControl` simply forwards `Time::outputTime()`); on the **NeoN**
backend the same keys are interpreted by the pure-Python policies above, giving
identical behaviour. Either way the case file is the standard OpenFOAM artefact.

### How the design follows SOLID

| Principle | In this design |
|-----------|----------------|
| **S** — Single Responsibility | `SolutionLoop` advances; `WriteControl` decides *when* to write; `Control` decides convergence/correctors; `TimeStepStrategy` decides `deltaT`; `RunTime` does primitive ops. One reason to change each. |
| **O** — Open/Closed | New write policy (`clockTime`), step strategy (PID), or control (`PisoControl`) is a new class implementing the protocol — the `SolutionLoop` and graph are not modified. |
| **L** — Liskov Substitution | Any `RunTime` (pybFoam `Time` ⇄ `NeoNRunTime`) and any `Control` (`Pimple` ⇄ `Simple`) is substitutable in `SolutionLoop` without changing its behaviour contract. |
| **I** — Interface Segregation | Each protocol is minimal: `WriteControl` exposes only `should_write`; `Control` only `converged`/`loop`; clients depend on nothing they don't use. |
| **D** — Dependency Inversion | `SolutionLoop` and `write_output` depend on the `Control` / `WriteControl` / `RunTime` *abstractions*; concrete backends and policies are injected at `Context` construction. |

The rest of this document details each abstraction (Props. 1–3) and grounds them
in the code that already exists.

---

## Supporting steady and unsteady in one loop structure

The central design question: how can a *single* loop structure run both a
transient (PISO/PIMPLE) and a steady (SIMPLE) case, the way OpenFOAM.org's one
`incompressibleFluid` solver does? The answer is that steady and unsteady differ
only in **data and injected behaviour**, never in **loop structure**.

### What is the same, what differs

| Aspect | Transient (PISO/PIMPLE) | Steady (SIMPLE) | Where it lives |
|--------|-------------------------|-----------------|----------------|
| Outer loop | `while runtime.run()` | `while runtime.run()` | **identical** — `SolutionLoop` |
| Meaning of "time" | physical time | iteration index | reinterpreted, same `RunTime` API |
| `endTime` | physical end time | max iteration count | `controlDict`, same key |
| `ddtSchemes` | `Euler`/`backward` | `steadyState` (ddt term → 0) | `fvSchemes`, **data** |
| Relaxation | none / light | `relaxationFactors` | `fvSolution`, **data** |
| Step size | CFL-adjusted | fixed | `TimeStepStrategy` (injected) |
| Outer correctors | `nOuterCorrectors` (PIMPLE) / 1 (PISO) | 1 | `Control` (injected) |
| Run termination | reaches `endTime` | residuals < `residualControl` | `Control.converged()` (injected) |

Everything in the "differs" column is either a dictionary value or an object
injected into the loop — **none of it is branching in the loop code**. That is
exactly what makes one structure serve both (OCP + DIP from the architecture
above).

### The single nested structure

The same graph expresses all three algorithms; the control supplies the loop
counts and the termination rule:

```
while control.run(runtime):           # OUTER: time steps (transient) | iterations (steady)
    time_step.adjust(ctx)             #   CourantTimeStep | FixedTimeStep   (injected)
    runtime.increment()
    while control.loop(ctx):          #   OUTER CORRECTORS: nOuterCorrectors | 1
        momentum_predictor()          #     under-relaxed in steady (data: relaxationFactors)
        while control.correct(ctx):   #     PRESSURE CORRECTORS: nCorrectors | 1
            while control.correctNonOrthogonal(ctx):
                pressure_solve()
        turbulence_correct()
    write_control.should_write() -> write()
```

Collapsing the counts recovers each classic algorithm:

- **PISO** — `nOuterCorrectors = 1`, `nCorrectors ≥ 2`, transient ddt → the outer
  corrector loop runs once per step.
- **PIMPLE** — `nOuterCorrectors ≥ 1`, transient ddt, optionally
  residual-driven outer correctors.
- **SIMPLE** — `nOuterCorrectors = 1`, `nCorrectors = 1`, `steadyState` ddt +
  relaxation; the run ends when `residualControl` is met.

This is precisely why OpenFOAM unifies them under `pimpleControl`: SIMPLE is
PIMPLE with a steady ddt scheme, one corrector, relaxation, and a
convergence-driven *outer* termination. The Python `PimpleControl`/`SimpleControl`
in `neofoam.algorithms.control` already encode these counts; the only addition
this proposal makes is `converged()`/`run()` (Prop. 2) so the *outer* run can stop.

### The simplification: the time loop *is* the main iteration loop

The whole thing becomes trivial once you stop thinking of "the time loop" and
"the SIMPLE iteration loop" as two things. **They are the same `while` loop.** In
OpenFOAM there is exactly one outer loop — `while (runTime.loop())` — and the
control wraps it (`simpleControl::loop()`, `pimpleControl::run(Time&)` both end in
`return runTime.loop()`). A steady run does not add a loop; it *reinterprets each
iteration of the existing loop* as a relaxation sweep instead of a physical step.

Consequences that make support nearly free:

- **No steady-only loop, no `converged() → end()` dance in the solver.** The
  outer predicate is just `control.run(runtime)`. For transient that is
  `runtime.run()`; for steady it ends the run on convergence first. One call.
- **The nested outer-corrector loop is optional.** It only materialises when
  `nOuterCorrectors > 1` (PIMPLE). For PISO and SIMPLE (`nOuterCorrectors == 1`)
  the main iteration loop *directly* drives one momentum + pressure solve per
  iteration — the structure is flat, exactly like hand-written `simpleFoam`.
- **"Time" needs no special-casing.** `endTime` is the iteration cap, `deltaT` is
  the (fixed) relaxation pseudo-step, `writeInterval` is an iteration interval —
  all the *same* `controlDict` keys and the *same* `RunTime` API, just
  reinterpreted. Nothing in the loop reads them differently.

So the design reduces to: **one iteration loop whose predicate is the control**,
plus injected `WriteControl`/`TimeStepStrategy`. Steady vs unsteady is a choice of
control + a `ddtScheme` — never a structural change.

### Subtlety: `residualControl` acts at different loop levels

`residualControl` appears in both, but means different things — and the **control
object encapsulates which level it governs**, so the loop never branches:

- **SIMPLE / steady** → governs the *outer run*: when satisfied,
  `SolutionControl.converged()` is true and `SolutionLoop` calls `runtime.stop()`.
  The simulation stops.
- **PIMPLE** → governs the *outer-corrector loop within a step*: when satisfied,
  `PimpleControl.loop()` returns false early and the solver proceeds to the next
  time step; the run still continues to `endTime`.

Same keyword, same residual source, two loop levels — resolved by
`SolutionControl.converged()` (outer run) vs `PimpleControl.loop()` (inner
correctors). The time loop is oblivious.

### Recommended approach (and the alternative rejected)

- **Recommended — one iteration loop, control as the predicate.** The outer
  predicate is `control.run(runtime)` (which reduces to `runtime.run()` for
  transient). Steadiness is injected via the `Control` (`run()`/`converged()` +
  corrector counts), the `TimeStepStrategy`, and the case's
  `ddtSchemes`/`relaxationFactors`. One graph, selected by config. Satisfies R3:
  the solver/loop contains no steady-state branch.
- **Rejected — two graphs (an `if steady:` fork).** Duplicates the structure,
  reintroduces the `algorithm_type == "SIMPLE"` smell, and breaks OCP: adding a
  third mode (e.g. transient-SIMPLE / pseudo-transient) would mean editing the
  loop. The injected-behaviour approach adds such a mode as a new `Control`
  subclass with no loop change.

### Residual wiring (the one piece to add)

For steady termination to work, the pressure/momentum solves must publish their
**initial residuals** into the context so `ResidualConvergenceCondition` can read
them (today `_extract_residual_from_context` returns `1.0`, i.e. "never
converged"). Both backends already produce them — pybFoam returns
`SolverPerformance`; NeoN's `PDESolver` returns solve statistics. The wiring:

```python
# in momentum / continuity operations, after each solve:
ctx.residuals["U"] = perf_U.initialResidual()      # pybFoam SolverPerformance
ctx.residuals["p"] = perf_p.initialResidual()

# SimpleControl reads them via the injected getter (replaces the 1.0 placeholder)
SimpleControl(..., residualControl={"p": 1e-3, "U": 1e-4})
#   _residual_check._get_residual = lambda ctx, f: ctx.residuals.get(f, 1.0)
```

With that single addition, the *same* loop structure runs steady and unsteady,
both backends, selected purely by the standard `controlDict` / `fvSolution` /
`fvSchemes` of the case.

---

## Context — what already exists in this repo

The Python `incompressibleFluid` solver
(`src/neofoam/solver/incompressibleFluid/incompressibleFluid.py`) already has a
time-loop abstraction built on the framework's operation graph. It is named
`TimeLoop` today; this proposal renames it to **`SolutionLoop`** (it is also the
steady iteration loop — see *The simplification* above) and changes its predicate
to delegate to the control (Prop. 2):

```python
class TimeLoop:                                 # → renamed SolutionLoop
    def __call__(self, ctx: Context) -> bool:
        return bool(ctx.runtime.run())          # the outer-loop predicate

...
with builder.loop(time_loop_op) as time_builder:        # IterativeOp(TimeLoop())
    time_builder.step(ops["set_time_step"])
    time_builder.step(ops["increment_time"])
    with time_builder.loop(algo_ops["inner_loop"]):     # PIMPLE inner loop
        inner_builder.step(algo_ops["momentum"])
        inner_builder.step(algo_ops["continuity"])
        inner_builder.step(ops["turbulence_correction"])
    time_builder.step(ops["write_output"])
```

Supporting pieces already present:

- `ctx.runtime` — currently a **pybFoam `Foam::Time`** (`run()`, `increment()`,
  `write()`, `timeName()`, `deltaTValue()`, `setDeltaT()`, `printExecutionTime()`).
- `neofoam.algorithms.control` — `PimpleControl` / `SimpleControl` (Pydantic),
  including `ResidualConvergenceCondition` and a `useResidualConvergence` mode on
  `SimpleControl`.
- `inner_loop(ctx) -> ctx.models["pimple_control"].loop()` — the inner corrector
  loop is *already* driven by the control object.

So this is **not a green-field design**. The proposal below is three focused
changes to make that existing loop (a) advance-only, (b) steady/transient via the
control algorithm, and (c) backend-agnostic.

## Requirements and how each is met

| # | Requirement | Met by |
|---|-------------|--------|
| **R1** | `controlDict` is the single source of truth | `ctx.runtime` already reads it; CFL params read from `controlDict` (Prop. 3) |
| **R2** | Single Responsibility Principle | advance ↔ converge ↔ step-size split across `RunTime` backend / control / CFL condition |
| **R3** | **The time loop only advances time** — the solver must not know about steady state | Prop. 2: `SolutionLoop` predicate is `control.run(runtime)`; convergence lives in the control, the `SIMPLE` branch is deleted from the solver (Prop. 3) |
| **R4** | Support **NeoN and pybFoam** backends | Prop. 1: `ctx.runtime` typed as a `RunTime` protocol both implement |

The guiding precedent is OpenFOAM's own `simpleControl::loop()` (read from
`.../solutionControl/simpleControl/simpleControl.C`):

```cpp
bool Foam::simpleControl::loop() {
    read();
    if (initialised_ && criteriaSatisfied())   // residuals converged?
        runTime.writeAndEnd();                  // ← the CONTROL ends the run
    else
        storePrevIterFields();
    return runTime.loop();                      // ← the TIME object only advances
}
```

The time object never decides steady vs transient; the *control* does, and in
OpenFOAM.org the single `incompressibleFluid` solver supports **both** modes
through one control + the `ddtScheme` (`steadyState` ⇒ steady). We mirror that.

---

## Proposal 1 — `RunTime` backend protocol (R4: NeoN + pybFoam)

`ctx.runtime` is the *only* object the loop touches that is backend-specific.
Promote it to a `Protocol`; pybFoam `Time` already satisfies most of it, and a
thin `NeoNRunTime` adapter wraps `NeoFOAM::RunTime`.

```python
# src/neofoam/framework/runtime.py
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class RunTime(Protocol):
    """Backend-agnostic time object. The loop talks only to this."""
    def run(self) -> bool: ...               # more steps to take? (pure advance test)
    def increment(self) -> None: ...         # ++runTime
    def end(self) -> None: ...               # writeAndEnd — used by the control on convergence
    def write(self, force: bool = False) -> None: ...
    def timeName(self) -> str: ...
    def deltaTValue(self) -> float: ...
    def setDeltaT(self, value: float) -> None: ...
    def printExecutionTime(self) -> None: ...
```

### pybFoam backend — already there, one method to bind

`pybFoam.Time` already implements `run/increment/write/timeName/deltaTValue/
setDeltaT/printExecutionTime`. Only `end()` (≙ `Foam::Time::writeAndEnd()`) needs
binding — it is what lets a steady run stop on convergence:

```cpp
// pybFoam: src/pybFoam/pybFoam_core/bind_time.cpp  (one line to add)
.def("end", &Foam::Time::writeAndEnd)
```

### NeoN backend — adapter over `NeoFOAM::RunTime`

```python
# src/neofoam/framework/runtime_neon.py
import neofoam as nf   # future NeoN bindings of NeoFOAM::RunTime


class NeoNRunTime:
    """Adapts NeoFOAM::RunTime (t, dt, controlDict) to the RunTime protocol."""
    def __init__(self, rt: "nf.RunTime") -> None:
        self._rt = rt
        self._end_time = float(rt.controlDict["endTime"])
        self._ended = False

    def run(self) -> bool:        return (not self._ended) and self._rt.t < self._end_time - 1e-10
    def increment(self) -> None:  self._rt.t = min(self._rt.t + self._rt.dt, self._end_time)
    def end(self) -> None:        self._ended = True
    def write(self, force: bool = False) -> None: nf.write_registered(self._rt, force)
    def timeName(self) -> str:    return f"{self._rt.t:g}"
    def deltaTValue(self) -> float: return self._rt.dt
    def setDeltaT(self, v: float) -> None:
        self._rt.dt = v
        nf.sync_run_times(self._rt, v)
    def printExecutionTime(self) -> None: nf.log.print_exec_time(self._rt)
```

`Context.runtime` is then typed `RunTime` (the protocol). The whole solver graph,
`SolutionLoop`, controls and hooks are unchanged across backends — selection happens
once, at `Context` construction (e.g. `--backend neon|pybfoam`).

---

## Proposal 2 — the time loop only advances; the control converges (R2, R3)

`SolutionLoop.__call__` stays a one-liner, but its predicate becomes
`control.run(ctx.runtime)` instead of bare `ctx.runtime.run()`. Steady-state
termination is **not** added to the loop itself — it is composed in via the active
control, mirroring `simpleControl::loop()`.

Add a `run(runtime)` to the control bundle that folds advance + termination into
the one predicate — the time loop and the steady iteration loop become the same
loop (see *The simplification* above). The control already owns the residual
machinery; we add `converged()` and the `run()` wrapper:

```python
# neofoam/algorithms/control.py  — add to PimpleControl / SimpleControl
class PimpleControl(BaseModel):
    def run(self, runtime: RunTime) -> bool:
        return runtime.run()                       # transient: advance to endTime

class SimpleControl(BaseModel):
    def converged(self) -> bool:
        assert self._residual_check is not None
        return self._residual_check.converged()    # residuals < residualControl

    def run(self, runtime: RunTime) -> bool:
        if self.converged():
            runtime.end()                          # == runTime.writeAndEnd()
        return runtime.run()                       # otherwise advance (iteration++)
```

```python
# src/neofoam/solver/incompressibleFluid/incompressibleFluid.py
class SolutionLoop:
    """The one main iteration loop. The RunTime advances; the control decides
    termination (cf. Foam::pimpleControl::run(Time&) / simpleControl::loop)."""
    def __init__(self, control: Any) -> None:
        self._control = control

    def __call__(self, ctx: Context) -> bool:
        return bool(self._control.run(ctx.runtime))   # advance + (steady) converge
```

Now:

- **Transient (PIMPLE):** `run()` is just `runtime.run()` ⇒ the run goes to
  `endTime`. Behaviourally identical to today.
- **Steady (SIMPLE):** `run()` consults `residualControl`; when satisfied, the
  control calls `runtime.end()` and the next `runtime.run()` returns `False`.
  `endTime` acts as the iteration cap. The loop structure did not change.

The solver no longer encodes "steady-state" anywhere — `SolutionLoop` just asks
the control whether to keep iterating. That is R3.

---

## Proposal 3 — one solver, both modes; delete the `SIMPLE` branch (R1, R3)

Today the solver leaks the steady/transient distinction into an operation:

```python
# CURRENT — set_time_step knows about SIMPLE (smell)
@incompressibleFluid.operation()
def set_time_step(self, ctx, pressure_velocity, cfl_condition):
    if getattr(pressure_velocity, "algorithm_type", "").upper() == "SIMPLE":
        return                       # ← solver branching on algorithm mode
    if cfl_condition is not None:
        cfl_condition(ctx)
```

CFL-based `deltaT` adjustment is a property of *transient time integration*, not
of the solver. Make it a `TimeStepStrategy` the control supplies, read from
`controlDict` (R1). A steady control supplies a no-op strategy; a transient one
supplies the Courant-limited strategy — so the solver stops branching:

```python
# neofoam/algorithms/control.py
class FixedTimeStep(BaseModel):
    def adjust(self, ctx: Any) -> None: ...                 # steady / fixed dt: nothing

class CourantTimeStep(BaseModel):
    maxCo: float; maxDeltaT: float
    def adjust(self, ctx: Any) -> None:
        co = ctx.compute_courant()                          # backend free fn
        rt = ctx.runtime
        if co > 1e-10:
            rt.setDeltaT(min(rt.deltaTValue() * min(self.maxCo / co, 1.2), self.maxDeltaT))

# control_factory.py — read the adjustTimeStep/maxCo/maxDeltaT keys from controlDict
def time_step_strategy(control_dict: dict) -> Any:
    if str(control_dict.get("adjustTimeStep", "no")).lower() == "yes":
        return CourantTimeStep(maxCo=float(control_dict["maxCo"]),
                               maxDeltaT=float(control_dict.get("maxDeltaT", "inf")))
    return FixedTimeStep()
```

```python
# CURRENT branch deleted — set_time_step becomes mode-agnostic
@incompressibleFluid.operation()
def set_time_step(self, ctx, time_step: Annotated[Any, "models"]) -> None:
    time_step.adjust(ctx)        # CourantTimeStep (transient) or FixedTimeStep (steady)
```

This is the OpenFOAM.org unification: a single `incompressibleFluid` solver runs
steady **and** transient. What changes between the two is purely configuration —
the `*Control` chosen from `fvSolution` (`PIMPLE` vs `SIMPLE`), `residualControl`,
and `ddtSchemes` (`steadyState` vs `Euler`/`backward`) — never the solver code.

---

## Worked example — same case & solver, both modes, both backends

```python
import neofoam as nf
from neofoam.solver import incompressibleFluid

# backend selected once; everything downstream is backend-agnostic (R4)
incompressibleFluid.run(["."], backend="pybfoam")   # or backend="neon"
```

Internally the graph is unchanged; only the injected control differs, and it is
chosen from `fvSolution`:

```python
# transient case: system/fvSolution has a PIMPLE dict, ddtSchemes default Euler
#   -> PimpleControl, converged()==False, CourantTimeStep  -> runs to endTime
#
# steady case:     system/fvSolution has a SIMPLE dict + residualControl,
#                  ddtSchemes steadyState
#   -> SimpleControl, converged() checks residuals, FixedTimeStep
#   -> SolutionLoop ends the run via runtime.end() when residuals drop below tol
```

The driver, the `SolutionLoop`, the inner `inner_loop`, `momentum`, `continuity`,
`write_output` operations are byte-for-byte identical in both cases and on both
backends.

---

## Why this satisfies SRP (R2)

| If this changes… | …only this changes |
|------------------|--------------------|
| backend (NeoN ⇄ pybFoam) | one `RunTime` impl (`NeoNRunTime` / pybFoam `Time`) |
| time advance / stop test | `SolutionLoop.__call__` (`control.run(runtime)`) |
| convergence criteria / steady vs transient | `SolutionControl.converged()` |
| `deltaT` policy | `TimeStepStrategy` (`Fixed` / `Courant`), built from `controlDict` |
| corrector counts | the control bundle (already the case) |

The time loop owns advancing time and nothing else; the control owns convergence
and corrector structure; the backend owns the primitive time ops. No operation in
the solver branches on the algorithm mode.

## Implementation checklist

> Step-by-step instructions with concrete diffs are in
> [`python-time-loop-implementation.md`](./python-time-loop-implementation.md).


1. **Prop. 1** — add `RunTime` protocol; bind `Time::writeAndEnd` as `.end()` in
   pybFoam; type `Context.runtime` as `RunTime`. (NeoN adapter lands with NeoN
   bindings.)
2. **Prop. 2** — rename `TimeLoop` → `SolutionLoop`; add a separate
   `SolutionControl` class with `run(runtime)` + `converged()` + `store_residual()`
   (not on `PimpleControl`/`SimpleControl`); predicate is
   `solution_control.run(ctx.runtime)` so it ends the run on convergence.
3. **Prop. 3** — add `FixedTimeStep`/`CourantTimeStep` strategies + factory from
   `controlDict`; delete the `SIMPLE` branch in `set_time_step`.
4. Re-enable `SIMPLE`/`PISO` in `pressure_velocity/base.py` (currently they fall
   back to PIMPLE) so a `SIMPLE` `fvSolution` actually selects `SimpleControl`.
5. Tests: `test/solver/incompressibleFluid` — assert a steady case ends on
   residual convergence before `endTime`, a transient case runs to `endTime`, and
   both produce identical graphs across backends (fake `RunTime`).
