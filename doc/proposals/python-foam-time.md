<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# Sketch: pure-Python `FoamTime` (time advancement matching OpenFOAM)

## Why

The time-advancement *logic* must live in **Python**, not be delegated to
`pybFoam.Time` / `NeoFOAM::RunTime`, so every backend shares one implementation
and behaves identically. It must match OpenFOAM's `Foam::Time` **exactly**,
including the **old-time-step bookkeeping** (`deltaT0`), and be verified against
the real pybFoam `Time`.

Backends then *use* this engine:

```
NeoNRunTime   ── owns ──▶ FoamTime (advancement)  +  NeoN field IO
PybFoamBackend── owns ──▶ FoamTime (advancement)  +  pybFoam field IO
                          (pybFoam.Time kept only as the parity oracle in tests)
```

## Interface

The advancement contract is an explicit interface so the `SolutionLoop`, write
control and step strategy depend on it — not on `FoamTime`, not on a backend
(DIP). `FoamTime` is one implementation; a backend may provide another.

```python
# src/neofoam/framework/time_interface.py
from typing import Protocol, runtime_checkable


@runtime_checkable
class TimeInterface(Protocol):
    """Time-advancement contract (the stateful clock — no field IO)."""
    def run(self) -> bool: ...           # value < endTime - 0.5*dt
    def end(self) -> bool: ...           # value > endTime + 0.5*dt  (OpenFOAM bool)
    def loop(self) -> bool: ...          # run() then increment()
    def increment(self) -> None: ...     # operator++ (advances + old-time update)
    def stop(self) -> None: ...          # writeAndEnd: terminate the run now
    def setDeltaT(self, dt: float, adjust: bool = True) -> None: ...
    def value(self) -> float: ...
    def deltaTValue(self) -> float: ...
    def deltaT0Value(self) -> float: ...  # old time step
    def timeIndex(self) -> int: ...
    def outputTime(self) -> bool: ...
```

`RunTime` (the solver-facing protocol in `framework/runtime.py`) is then
**`TimeInterface` + field IO** — it extends the advancement contract with the
backend-only methods:

```python
@runtime_checkable
class RunTime(TimeInterface, Protocol):
    def write(self, force: bool = False) -> None: ...
    def printExecutionTime(self) -> None: ...
    def timeName(self) -> str: ...
    @property
    def controlDict(self) -> Any: ...
```

So advancement is shared (pure Python, `FoamTime`); only IO differs per backend.
`pybFoam.Time` already satisfies the whole of `RunTime` structurally, which is
what makes the parity test possible.

### Why `setDeltaT` + `increment`, not `advance(deltaT)`

A single `advance(deltaT)` looks tidier, but `deltaT` is **state read by the
whole step**, not just an argument to the advance — so the two-method shape earns
its keep:

1. **`deltaT` and `deltaT0` are read by consumers *during* the step.** `ddt`
   schemes (`backward`, `CrankNicolson`) need both the current and the *old* step
   size; CourantNo, boundary conditions and functionObjects read `deltaTValue()`.
   It must be queryable state, set once and read many times — not hidden inside an
   advance call.
2. **"Set" is not a plain store — it can adjust.** `setDeltaT(dt, adjust=True)`
   runs `adjustDeltaT()` (the `adjustableRunTime` snap to land exactly on the
   write time, clamped to ×2 / ×0.2). That is a real operation distinct from
   advancing.
3. **`deltaT` changes that are *not* advances:** the initial value from
   `controlDict`, a `runTimeModifiable` re-read mid-run, sub-cycle save/restore.
   You set without stepping.
4. **Old-time correctness.** `increment` is the single commit point that rolls
   `deltaT0 = deltaTSave; deltaTSave = deltaT` and decides `writeTime`. Keeping
   the roll in `increment` makes it happen exactly once per real step, regardless
   of how many times `setDeltaT` was (or wasn't) called.
5. **Exact parity.** Matching `Foam::Time` 1:1 (a hard requirement) means our
   `increment` must reproduce `operator++`, which is independent of `setDeltaT`.
   Mirroring the two-method shape keeps parity obvious.
6. **SRP / DIP fit.** "Decide the next step size" is the `TimeStepStrategy`'s job
   (`set_time_step` op → `setDeltaT`); "commit the step" is the loop's job
   (`increment_time` op → `increment`). One `advance(dt)` would re-couple them.

If a call site genuinely wants the one-liner, add it as **sugar** over the
primitives — a **call operator** reads naturally and mirrors C++ `operator++`:

```python
def __call__(self, delta_t: float | None = None) -> None:
    """``time()`` advances one step; ``time(dt)`` sets the step then advances."""
    if delta_t is not None:
        self.setDeltaT(delta_t)
    self.increment()
```

`setDeltaT` / `increment` stay the primitives (state-reads, adjust, old-time roll,
parity); `__call__` is just the convenient surface.

## Delivered as a `ModelSpec`

The time is a **`ModelSpec`** (like the pressure-velocity `Model`), not a loose
class. This is exactly the config/state split the framework already gives us:

- `@solutionLoop.load(case_dir, instance_id) -> ControlDictConfig` — the **config**
  (a validated `BaseConfig`, loaded from `system/controlDict`).
- `spec.instantiate(...) -> ModelRuntime` — the mutable **state** holder
  (`ModelRuntime.config` is the `ControlDictConfig`).
- `@solutionLoop.build(config) -> [InitStep]` — lazily creates the `FoamTime` clock
  from the validated config and exposes it as a model the loop can reach.
- `@solutionLoop.operation(...)` — the time graph steps (`set_time_step`,
  `increment_time`, `write_output`).

```python
# src/neofoam/framework/time/spec.py
from neofoam.framework.model import ModelSpec
from neofoam.framework.initialization import model

solutionLoop = ModelSpec("solutionLoop")


@solutionLoop.load
def load(case_dir: Path, instance_id: str) -> ControlDictConfig:
    return ControlDictConfig.load(case_dir=case_dir)        # controlDict is the SSOT


@solutionLoop.build
def build(config: ControlDictConfig) -> list[InitStep]:
    # mutable clock + injected policies, all from the validated config
    mode = advance_mode_from_schemes(config)              # AdvanceMode.TIME | ITERATION
    return [
        # the clock is a TimeInterface — FoamTime is the default impl, swappable
        model("clock",        lambda ctx: make_clock(config, mode)),
        model("write_control", lambda ctx: write_control_from_dict(config, config.startTime)),
        # config-owned deltaT constraints; other models inject their own (see below)
        *[model(f"dtc_{i}", lambda ctx, c=c: c)
          for i, c in enumerate(delta_t_constraints_from(mode, config))],
    ]


@solutionLoop.operation()                                   # was incompressibleFluid.set_time_step
def set_time_step(self, ctx, time_step: Annotated[Any, "models"]) -> None:
    time_step.adjust(ctx)                              # clock.setDeltaT under the hood

@solutionLoop.operation()
def increment_time(self, ctx, clock: Annotated[TimeInterface, "models"]) -> None:
    Info(f"Time = {clock.value()}")
    clock.increment()                                  # operator++ (old-time update)

@solutionLoop.operation(depends_on=["turbulence_correction"])
def write_output(self, ctx, clock: Annotated[TimeInterface, "models"],
                 write_control: Annotated[Any, "models"]) -> None:
    if write_control.should_write(clock):
        clock.write()                                  # IO delegated to the backend runtime
    clock.printExecutionTime()
```

The clock is built through a **factory seam**, never `FoamTime` directly, so the
implementation is swappable (DIP):

```python
def make_clock(config, mode, *, impl: type[TimeInterface] = FoamTime,
               port: TimeBackendPort | None = None) -> TimeInterface:
    return impl.from_config(config, mode, port)        # FoamTime today; another backend later
```

`ControlDictConfig(BaseConfig)` already exists
(`solver/incompressibleFluid/configs.py`, `@IOStrategy(OF("system/controlDict"))`)
validating `endTime`/`deltaT`/`adjustTimeStep`/`maxCo`/`writeControl`/`writeInterval`;
this change just **extends** it with `startTime` and `maxDeltaT`.

### Config vs state — `ModelSpec`/`ModelRuntime`, *not* OpenFOAM's layout

The `ModelSpec` ⇄ `ModelRuntime` boundary **is** the config/state split:

- **Config** = `ModelRuntime.config` → `ControlDictConfig(BaseConfig)`: validated,
  immutable, IO-aware (from `@load`).
- **State** = the mutable clock the engine advances (`value`, `timeIndex`,
  `deltaT`, `deltaT0`/`deltaTSave`, `writeTime`, `writeTimeIndex`) — a
  **`TimeInterface`** built in `@build`. `FoamTime` is the *default* implementation,
  **not** the type the rest of the code depends on.

`FoamTime` is just one swappable backend behind `TimeInterface` — the `@build`
factory (`make_clock`) can return a different implementation later (e.g. a
backend-native clock) with no change to `SolutionLoop`, the operations, the
write control or the step strategy, which all depend on the interface (DIP). So
"the time is `FoamTime`" is true only *today* and only *by default*.

OpenFOAM bundles config + state across a `TimePaths → TimeState → Time`
inheritance chain. **We deliberately do not replicate that hierarchy** — the
framework's spec/runtime split already separates the concerns, and we only have
to match *behaviour* (the formulas below). The backend's runtime carries its own
state (handles, registry, mesh) and merely *composes* a `TimeInterface` rather
than inheriting any shape.

## Time-based vs iteration-based — injected, not branched

Whether a step means "advance physical time" (transient) or "do one iteration"
(steady) is an **injected policy**, exactly like the control and the step
strategy — the loop and `FoamTime` never branch on it. The advancement formula is
identical (`value += deltaT`; `run() = value < endTime − 0.5·deltaT`); the *mode*
only supplies the differing pieces:

The set is closed — there are exactly two modes — so this is an **`Enum`**, not a
strategy hierarchy:

```python
class AdvanceMode(Enum):
    """Injected (exactly two): is a step physical time or an iteration?"""
    TIME = "time"            # transient: advance physical time, deltaT-sized
    ITERATION = "iteration"  # steady: unit step, "time" is the iteration index
```

The two-way difference is small: the mode decides whether any `deltaT`
constraints are injected at all, and how time is named.

```python
def delta_t_constraints_from(mode: AdvanceMode, config: ControlDictConfig) -> list[DeltaTConstraint]:
    if mode is AdvanceMode.ITERATION or not config.adjustTimeStep:
        return []                                         # deltaT stays fixed
    return [MaxDeltaTConstraint(maxDeltaT=config.maxDeltaT)]   # flow model adds CourantConstraint
```

(Step sizing itself is the injectable-constraints pipeline below; the mode only
gates it.) `FoamTime.from_config(config, mode)` uses the enum for the only two
things that
differ: the initial step (`deltaT = 1` for `ITERATION`) and `timeName` (integer
vs general float). Selection is from the case — `ddtSchemes steadyState ⇒
ITERATION` — via `advance_mode_from_schemes(config)`, injected in the
`solutionLoop` `@build` shown above.

Why an `Enum` here but strategy *classes* for `WriteControl` / `TimeStepStrategy`
/ control? Those families are **open** (a new `clockTime` write policy, a PID
step controller, a `PisoControl` are all plausible new subclasses). The advance
mode is **closed** — there is no third kind of "step" — so a hierarchy would be
over-engineering; one enum + one dispatch is clearer and still injected.

So the four orthogonal choices compose:

| | step sizing | early stop |
|---|---|---|
| **transient** | `AdvanceMode.TIME` — `min` of injected `DeltaTConstraint`s | none — runs to `endTime` (`PimpleControl`) |
| **steady** | `AdvanceMode.ITERATION` (unit step, no constraints) | residual convergence (`SimpleControl`) |

`FoamTime`, `SolutionLoop`, `WriteControl` are identical across all four cells —
only the injected `AdvanceMode` + control differ.

## Injectable stability criteria → `deltaT`

The next `deltaT` is **not** a single hardwired Courant number — it is the
**minimum over every stability criterion that models inject**, then the
adjustable-runtime write alignment. This mirrors OpenFOAM, where solvers stack
constraints (`maxCo`, VoF's `maxAlphaCo`, a reaction/`maxDeltaT`, functionObjects'
`adjustTimeStep()`); each is a contribution, the controlling one wins.

A criterion is a tiny injectable interface — any model can provide one:

```python
@runtime_checkable
class DeltaTConstraint(Protocol):
    """The largest deltaT a model permits for the next step (VGREAT = no limit)."""
    def max_delta_t(self, ctx: Any) -> float: ...


class CourantConstraint(BaseModel):           # contributed by the flow model
    maxCo: float
    def max_delta_t(self, ctx) -> float:
        co = ctx.max_courant()                # the raw scalar via the backend port
        dt = ctx.models["clock"].deltaTValue()
        return dt * (self.maxCo / co) if co > SMALL else VGREAT

class InterfaceCourantConstraint(BaseModel):  # contributed by a VoF model (maxAlphaCo)
    maxAlphaCo: float
    def max_delta_t(self, ctx) -> float: ...

class MaxDeltaTConstraint(BaseModel):         # from controlDict
    maxDeltaT: float
    def max_delta_t(self, ctx) -> float: return self.maxDeltaT
```

`set_time_step` aggregates **all** injected constraints (min), clamps the growth,
then `setDeltaT(dt, adjust=True)` applies the **adjustableRunTime** snap (the final
composable stage — it lands the step exactly on the next write time):

```python
@solutionLoop.operation()
def set_time_step(self, ctx, clock: Annotated[TimeInterface, "models"],
                  constraints: Annotated[list[DeltaTConstraint], "models"]) -> None:
    if not constraints:                       # iteration mode / adjustTimeStep no
        return
    allowed = min(c.max_delta_t(ctx) for c in constraints)     # every injected criterion
    allowed = min(allowed, 1.2 * clock.deltaTValue())          # growth clamp
    clock.setDeltaT(allowed, adjust=True)                      # adjust ⇒ write-time snap
```

So both halves are injectable/composable:

- **stability criteria** — any model registers a `DeltaTConstraint` (the flow
  model contributes `CourantConstraint`, a VoF model adds
  `InterfaceCourantConstraint`, …); the loop just takes the `min`.
- **adjustable runtime** — the write-time alignment is the `adjust=True` stage of
  `setDeltaT` (`FoamTime.adjustDeltaT`, only active for `adjustableRunTime`),
  applied after the criteria.

`AdvanceMode.ITERATION` and `adjustTimeStep no` simply inject **no** constraints,
so `deltaT` is left fixed — no branching in the loop. This replaces the earlier
single `CourantTimeStep` strategy: the Courant limit is now just *one* injected
constraint among many.

## Maximising the Python core — the minimal backend port

Goal: keep **as much logic as possible in Python**, so the backend shrinks to a
tiny *port* of operations that are genuinely impossible in pure Python (they
touch C++ field data, the file system, or MPI). Everything that is a *decision*,
a *number*, or a *string* lives in the Python core (`FoamTime` + the
`solutionLoop` `ModelSpec`) and is therefore shared and parity-tested.

Apply the test: **"does this need a field / the mesh / the file system / a
parallel reduce?"** If no → Python.

| Logic | needs backend? | Home |
|-------|:--------------:|------|
| advance math (`run`/`loop`/`increment`/`end`, `value`/`deltaT`/**`deltaT0`**/`timeIndex`) | no | **Python** `FoamTime` |
| write **decision** (`writeControl`/`writeInterval`) | no | **Python** `WriteControl` |
| `deltaT` **adjust formula** (Courant scale, ×2/×0.2 clamp, `adjustableRunTime`) | no | **Python** `TimeStepStrategy`/`FoamTime.adjustDeltaT` |
| stop logic (`writeAndEnd`, `stopAt`, end-time snap) | no | **Python** `FoamTime` |
| **time-name formatting** (general/fixed/scientific + precision bump) | no (pure string/number) | **Python** `FoamTime.timeName` |
| **runTimeModifiable** controlDict re-read | no (text → `ControlDictConfig.load`) | **Python** (re-`@load`) |
| **execution-time** reporting (wall clock) | no (`time.perf_counter`) | **Python** |
| Courant **number** (needs `phi`, mesh `deltaCoeffs`) | yes | **port** `courant_number()` → scalar |
| **persist fields** to disk | yes | **port** `write_fields(time_name)` |
| parallel **reduce** (max across ranks) | yes (serial = identity) | **port** `reduce_max(x)` |

So the irreducible backend surface is ~3 methods:

```python
# src/neofoam/framework/time/backend_port.py
class TimeBackendPort(Protocol):
    """The only things FoamTime cannot do in pure Python."""
    def write_fields(self, time_name: str) -> None: ...      # persist registered fields
    def courant_number(self) -> float: ...                   # max Co (0.0 if no flux yet)
    def reduce_max(self, value: float) -> float: ...          # MPI max; serial = value
```

`FoamTime` owns the clock and calls the port only for those three. This is the
"ports & adapters" shape: a fat Python core, a thin adapter per backend.

```python
class FoamTime(TimeInterface):
    def __init__(self, cfg: ControlDictConfig, port: TimeBackendPort) -> None: ...
    # increment(), adjustDeltaT(), timeName(), outputTime(), stop(), run() ... all Python
    def write(self) -> None:                 # decision in Python, bytes via the port
        self._port.write_fields(self.timeName())
    def max_courant(self) -> float:
        return self._port.courant_number()   # only the raw scalar crosses the boundary
```

### Consequence for the `RunTime` split

This **shrinks** the backend half of `RunTime`: `timeName`,
`printExecutionTime`, controlDict re-read and the write *decision* move into the
Python core. What truly differs per backend collapses to the 3-method
`TimeBackendPort` (+ optionally a backend-selected `solutionLoop` `ModelSpec` via
`@detect` / `PluginSystem` for cases a backend must specialise). The
`SolutionLoop`, write control and step strategy never change.

The parity test (below) asserts equality **only** on the shared advancement
quantities and the (now Python) `timeName`; it does not compare `write_fields`,
which is the one place behaviour is meant to diverge.

## Reference: `$FOAM_SRC/OpenFOAM/db/Time` (v2406)

Extracted verbatim from `Time.C` / `TimeState.{H,C}` — the exact rules to mirror.

**State (`TimeState`)**, all initialised to `0`/`false`:
`value` (time), `timeIndex_`, `writeTimeIndex_`, `deltaT_`, `deltaT0_`,
`deltaTSave_`, `deltaTchanged_`, `writeTime_`.

**`run()`** (const query):
```
isRunning = value < endTime_ - 0.5*deltaT_
```

**`loop()`**:
```
isRunning = run();  if (isRunning) ++(*this);  return isRunning
```

**`end()`** (const query — note: a *bool*, NOT "terminate"):
```
return value > endTime_ + 0.5*deltaT_
```

**`operator++()`** (the core advance):
```
deltaT0_    = deltaTSave_          # ← OLD time-step update
deltaTSave_ = deltaT_
oldTimeValue = value
setTime(value + deltaT_, timeIndex_ + 1)     # value += dt; index += 1

if |value| < 10*SMALL*deltaT_:  setTime(0.0, timeIndex_)   # snap to zero

writeTime_ = false
switch writeControl_:
  wcTimeStep:           writeTime_ = (timeIndex_ % int(writeInterval_) == 0)
  wcRunTime |
  wcAdjustableRunTime:  writeIndex = int(((value - startTime_) + 0.5*deltaT_)/writeInterval_)
                        if writeIndex > writeTimeIndex_:
                            writeTime_ = true;  writeTimeIndex_ = writeIndex
  wcCpuTime/wcClockTime: (wall-clock based — omitted, non-deterministic)

# stop-at handling (writeAndEnd etc.)
if not end():
  if stopAt_ == saNoWriteNow:  endTime_ = value
  elif stopAt_ == saWriteNow:  endTime_ = value; writeTime_ = true
  elif stopAt_ == saNextWrite and writeTime_: endTime_ = value
# (writeOnce, signals, time-name precision bump: omitted — see below)
```

**`setDeltaT(dt, adjust=true)`**:
```
deltaT_ = dt;  deltaTchanged_ = true;  if adjust: adjustDeltaT()
```

**`adjustDeltaT()`** (only bites for `adjustableRunTime`):
```
if writeControl_ == wcAdjustableRunTime:
    timeToNextWrite = max(0, (writeTimeIndex_+1)*writeInterval_ - (value - startTime_))
    nSteps = timeToNextWrite / deltaT_
    if nSteps < labelMax:
        nStepsToNextWrite = max(1, round(nSteps))         # round = half away from zero
        newDeltaT = timeToNextWrite / nStepsToNextWrite
        if newDeltaT >= deltaT_:  deltaT_ = min(newDeltaT, 2.0*deltaT_)   # grow ≤ ×2
        else:                     deltaT_ = max(newDeltaT, 0.2*deltaT_)   # shrink ≥ ×0.2
```

Constants: `SMALL = 1e-15`, `labelMax = 2**31 - 1`. `int(...)`/`label(...)`
truncates toward zero; `round(...)` is half-away-from-zero (≠ Python `round`).

## Proposed `src/neofoam/framework/foam_time.py`

```python
class FoamTime(TimeInterface):
    """Pure-Python re-implementation of Foam::Time's advancement.

    Backend-agnostic: holds only the mutable clock + writeControl. Field IO
    (write / printExecutionTime) is the backend's job and lives elsewhere.
    Configured by a validated BaseConfig, not loose kwargs.
    """
    @classmethod
    def from_config(cls, cfg: ControlDictConfig) -> "FoamTime": ...

    # --- TimeInterface: queries (names mirror pybFoam.Time) ---
    def value(self) -> float          # current time
    def deltaTValue(self) -> float
    def deltaT0Value(self) -> float   # ← old time step
    def timeIndex(self) -> int
    def run(self) -> bool             # value < endTime - 0.5*dt
    def end(self) -> bool             # value > endTime + 0.5*dt   (OpenFOAM bool)
    def outputTime(self) -> bool      # == writeTime_

    # --- TimeInterface: mutators ---
    def setDeltaT(self, dt, adjust=True) -> None
    def increment(self) -> None       # operator++  (does the deltaT0 update etc.)
    def loop(self) -> bool            # run() then increment()
    def stop(self) -> None            # writeAndEnd: end the run now
```

### Naming fix (carry into the existing code)

OpenFOAM's `end()` is the *bool* "are we past endTime", so the **terminate**
method introduced earlier is renamed `end()` → **`stop()`** across
`framework/runtime.py` (the `RunTime` protocol), `algorithms/control.py`
(`SolutionControl.run` calls `runtime.stop()`), and the existing tests/fakes.
`stop()` sets `endTime_ = value` (so the next `run()` is `False`) and
`writeTime_ = True`, replicating `Foam::Time::writeAndEnd()`.

### What is intentionally omitted (and why it doesn't affect parity)

functionObjects, MPI signals, profiling, `writeOnce`, and CPU/clock write
control — none change the numeric `value`/`deltaT`/`deltaT0`/`timeIndex`/
`writeTime` sequence of a normal run.

**Kept in Python** (per "maximise the core"): `timeName()` formatting
(`general`/`fixed`/`scientific` + `timePrecision`) and the precision-bump that
disambiguates colliding time names — both are pure string/number logic, so they
live in `FoamTime` and are parity-tested against `pybFoam.Time.timeName()`.

## Test plan (TDD)

1. **`test/framework/test_foam_time.py` — the spec (pure, no bindings).**
   Encodes the rules above directly:
   - fixed `deltaT`: value sequence `dt,2dt,…`, stops at `endTime` via the
     `endTime − 0.5·dt` rule; `timeIndex` increments 1..N.
   - **old time**: after each `increment`, `deltaT0Value()` equals the
     *previous* `deltaT` (and `0` on the first step).
   - `writeControl timeStep` + `writeInterval=2` → `outputTime()` true on even
     indices; `runTime`/`adjustable` + `writeInterval=0.2` → true at the right
     sim-times via the `((value-start)+0.5dt)/interval` index rule.
   - `setDeltaT` mid-run: subsequent values reflect the new step; `deltaT0`
     still tracks correctly across the change.
   - `adjustableRunTime` + `setDeltaT(adjust=True)`: `deltaT` snapped so an
     integer number of steps lands on the write time, clamped to ×2 / ×0.2.
   - `stop()` → next `run()` is `False` and `outputTime()` is `True`.

2. **`test/framework/test_foam_time_parity.py` — match pybFoam exactly.**
   `pytest.importorskip("pybFoam")`; build a real `pybFoam.Time` from a temp
   case (write a minimal `system/controlDict`), build a `FoamTime` from the same
   keys, then drive both with the identical `while loop():` and assert per step:
   ```
   (value(), deltaTValue(), timeIndex(), outputTime())  are equal
   ```
   for `writeControl ∈ {timeStep, runTime, adjustableRunTime}` and with a CFL-style
   `setDeltaT` applied each step. Skips cleanly when the OpenFOAM environment
   needed to construct `Time` is absent.

## Integration (follow-up, not this change)

`NeoNRunTime` / the pybFoam backend hold a `FoamTime` for advancement and add
field IO; `SolutionLoop`'s predicate (`solution_control.run(ctx.runtime)`) is
unchanged. This makes the time logic identical on every backend by construction.

> **`SolutionControl` is its own class** (`algorithms/control.py`), separate from
> the algorithm controls. It owns the outer-loop predicate `run(runtime)` +
> `converged()` + `store_residual()` (empty `residualControl` ⇒ transient, set ⇒
> steady). `PimpleControl` / `SimpleControl` keep *only* the corrector-loop
> structure. The `SolutionLoop` delegates to the `SolutionControl`, not to an
> algorithm control.
