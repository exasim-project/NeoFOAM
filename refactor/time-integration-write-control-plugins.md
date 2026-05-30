<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# Refactor: time integration & write control on `BaseConfig` + `PluginSystem`

> **Status (implemented):** done, with two changes from this plan: (1) the whole
> time-loop stack lives under **`src/neofoam/algorithms/`** (not `framework/`) —
> `foam_time.py`, `time_integration.py`, `write_control.py`, `writer.py`,
> `solution_loop.py` (engine **and** `solutionLoop` Model merged), `field_writer.py`
> (`fieldWriter` Model), beside `control.py`/`time_step.py`; (2) the solver registers
> these as core Models and injects pybFoam adapters. Dead/duplicate helpers removed:
> `SolutionLoop.from_config`, `delta_t_constraints_from`, `make_write_control`.
> Paths below say `framework/…`; read them as `algorithms/…`.

## Goal

Make **time integration** (the stepper / advancement strategy) and **write
control** (when to persist) first-class members of the existing extensibility
machinery the rest of the time-loop already uses:

- every selectable behaviour is a **`PluginSystem`** family — a new variant is a
  new `@Base.register` class, never an edit to an `if`/`enum` branch (OCP);
- every selectable behaviour is parameterised by a validated, file-backed
  **`BaseConfig`** — the case file is the single source of truth, and a plugin
  extends the schema by subclassing the config.

The result removes the two places that still bypass this: the `AdvanceMode`
enum branched *inside* `FoamTime`, and the hand-written `write_control_from_config`
factory branch.

## Current state (audit)

| Concern | Type today | Uses `PluginSystem`? | Uses `BaseConfig`? | Action |
|---|---|---|---|---|
| `WriteControl` (when to write) | `PluginSystem` family in `write_control.py` | ✅ | ✅ (`WriteControlConfig`) | **keep**; fix *selection* (below) |
| `DeltaTConstraint` (deltaT limits) | `PluginSystem` family in `time_step.py` | ✅ | ✅ | keep |
| `Writer` / `FieldHook` (how to write) | `PluginSystem` families in `writer.py` | ✅ | ✅ | keep |
| **`FoamTime` stepper + `AdvanceMode`** | plain class + `Enum`, branched internally | ❌ | partial (`TimeControlConfig`) | **refactor → `TimeIntegration` family** |
| `SolutionLoop` engine | plain class | ❌ | — | keep as engine; construct from plugins |
| `SolutionControl` / `Pimple` / `Simple` | plain `pydantic.BaseModel` | ❌ | ❌ | **secondary**: elevate to family + `BaseConfig` |
| Write-control *selection* | `write_control_from_config` `if` branch | ❌ (bypasses the union) | reads `BaseConfig` | **refactor → load via discriminated union** |

Two things already work and must **not** be redone: the `WriteControl` plugin
family/policies, and the `DeltaTConstraint` family. The gaps are the *stepper*
and the way the `WriteControl` policy is *selected*.

## Problem 1 — `FoamTime` branches on an `AdvanceMode` enum

`foam_time.py` hard-codes the two advancement regimes as an `Enum` and branches
on it in four places:

```python
self._delta_t = 1.0 if mode is AdvanceMode.ITERATION else delta_t   # __init__
if self._mode is AdvanceMode.ITERATION: return str(int(round(...)))  # timeName
if mode is AdvanceMode.ITERATION or not config.adjustTimeStep: ...    # delta_t_constraints_from
```

Adding a third regime (pseudo-transient, local-time-stepping, dual-time) means
editing `FoamTime` — an OCP violation, and the one spot in the time-loop stack
that is neither a plugin nor config-selected.

### Target — a `TimeIntegration` `PluginSystem` family

`TimeIntegration` becomes the plugin interface; `FoamTime` keeps the shared
bookkeeping (`_value`/`_index`/`deltaT0`, the `StepSink` push, `run`/`end`) but
delegates the regime-specific decisions to the injected integration:

```python
# src/neofoam/framework/time_integration.py
from typing import Any, Literal
from pydantic import Field
from neofoam.core.plugin_system import PluginSystem
from neofoam.io import BaseConfig
from neofoam.algorithms.time_step import DeltaTConstraint, MaxDeltaTConstraint, VGREAT


@PluginSystem.register(
    discriminator_variable="integration", discriminator="time_integration_type"
)
class TimeIntegration(BaseConfig):
    """Plugin interface: how a step advances (physical time vs. iteration).

    A plugin owns the regime-specific decisions the stepper used to branch on:
    the initial deltaT, how a step name renders, and which deltaT constraints
    apply. New regimes register with @TimeIntegration.register.
    """

    def initial_delta_t(self, requested: float) -> float:
        raise NotImplementedError

    def step_name(self, value: float, index: int, precision: int) -> str:
        raise NotImplementedError

    def constraints(self, config: "TimeControlConfig") -> list[DeltaTConstraint]:
        raise NotImplementedError


@TimeIntegration.register
class TransientIntegration(TimeIntegration):
    """ddtScheme Euler/backward/CrankNicolson — advance physical time, deltaT-sized."""

    time_integration_type: Literal["transient"] = "transient"

    def initial_delta_t(self, requested: float) -> float:
        return requested

    def step_name(self, value: float, index: int, precision: int) -> str:
        return f"{value:.{precision}g}"

    def constraints(self, config: "TimeControlConfig") -> list[DeltaTConstraint]:
        if not config.adjustTimeStep:
            return []
        cap = config.maxDeltaT if config.maxDeltaT is not None else VGREAT
        return [MaxDeltaTConstraint(maxDeltaT=cap)]


@TimeIntegration.register
class SteadyIntegration(TimeIntegration):
    """ddtScheme steadyState — unit step; "time" is the iteration index, fixed dt."""

    time_integration_type: Literal["steady"] = "steady"

    def initial_delta_t(self, requested: float) -> float:
        return 1.0

    def step_name(self, value: float, index: int, precision: int) -> str:
        return str(int(round(value)))

    def constraints(self, config: "TimeControlConfig") -> list[DeltaTConstraint]:
        return []   # fixed pseudo-step
```

`FoamTime` then holds a `TimeIntegration` instead of an `AdvanceMode`:

```python
class FoamTime:
    def __init__(self, *, integration: TimeIntegration, delta_t: float, ...):
        self._integration = integration
        self._delta_t = integration.initial_delta_t(delta_t)
        ...

    def timeName(self) -> str:
        return self._integration.step_name(self._value, self._index, self._precision)
```

`delta_t_constraints_from(mode, config)` collapses to
`integration.constraints(config)`; `advance_mode_from_ddt(ddt_default)` becomes a
selector that returns the *plugin* (or its discriminator string) instead of the
enum:

```python
def integration_from_ddt(ddt_default: str) -> TimeIntegration:
    kind = "steady" if ddt_default.strip() == "steadyState" else "transient"
    return TimeIntegration.create(integration={"time_integration_type": kind}).integration
```

> `AdvanceMode` (enum), the two-arg `delta_t_constraints_from`, and the
> `mode is AdvanceMode.ITERATION` branches are **deleted**.

## Problem 2 — write-control selection bypasses the discriminated union

`write_control.py` already has the `WriteControl` family right, but
`write_control_from_config` re-implements selection with a Python `if`:

```python
def write_control_from_config(config, *, stepper_decides=False):
    if stepper_decides:
        return StepperWriteControl()
    if config.writeControl == "timeStep":
        return IntervalWriteControl(interval=int(config.writeInterval))
    return RunTimeWriteControl(interval=float(config.writeInterval), start=config.startTime)
```

A new policy (`clockTime`, `cpuTime`) means editing this branch — the registry
exists but the selection doesn't use it. The discriminator should be driven by
the config keyword so registration alone is enough.

### Target — load the policy through `WriteControl.plugin_model`

Validate the policy from a mapping whose discriminator (`write_control_type`) is
filled from the controlDict `writeControl` keyword, so the discriminated union
picks the registered class:

```python
def write_control_from_config(config, *, stepper_decides=False):
    if stepper_decides:
        return StepperWriteControl()
    payload = _policy_payload(config)         # {"write_control_type": "timeStep", "interval": ...}
    return WriteControl.create(policy=payload).policy
```

`_policy_payload` maps the controlDict keys to a policy's fields (the only
keyword-specific glue left, and it lives next to the config). Each policy keeps
declaring its own `write_control_type` literal; adding `clockTime` is then a new
`@WriteControl.register` class plus one mapping entry — no edit to a behavioural
branch.

> Optional stretch: give `WriteControlConfig` a `to_policy_payload()` method so
> even the mapping lives on the config (schema-owned), making selection pure.

## Problem 3 (secondary) — controls are plain `BaseModel`

`SolutionControl` / `PimpleControl` / `SimpleControl` in `algorithms/control.py`
are plain `pydantic.BaseModel`, and `LoopControl` is a bare `Protocol`. For full
consistency the outer-loop predicate should be a `PluginSystem` family
(`LoopControl` family with `transient`/`steady` registrations) on `BaseConfig`,
selected from `fvSolution`. This is **lower priority** — it is orthogonal to the
two requirements above and larger in blast radius (residual wiring, the
corrector bundles). Recommend scoping it as a follow-up unless explicitly
wanted now.

## Wiring (solver models)

The two solver-side Models change only their construction; their operations are
untouched.

`models/solution_loop.py`:

The `ddtSchemes` default is an `fvSchemes` value, not a `controlDict` key — today
`advance_mode_from_ddt(ddt_default: str)` already takes it as a string and the
solver build defaults to `AdvanceMode.TIME` (the ddt→mode wiring is not yet
connected). This refactor keeps that seam: `integration_from_ddt(ddt_default)`
replaces `advance_mode_from_ddt`, and connecting the real `fvSchemes` read stays
a separate concern.

```python
@solutionLoop.build
def build(config: ControlDictConfig) -> list[InitStep]:
    integration = integration_from_ddt(ddt_default)          # ddt_default read from fvSchemes
    def create_stepper(ctx):
        return FoamTime.from_config(config, integration=integration,
                                    sink=PybFoamStepSink(ctx["runtime"]))
    def create_engine(ctx):
        loop = SolutionLoop(stepper=ctx["models.stepper"], control=SolutionControl())
        for c in integration.constraints(config):
            loop.add_constraint(c)
        ...
```

`models/field_writer.py` is unchanged — `write_control_from_config` keeps the
same signature; only its body now goes through the union.

`FoamTime.from_config(...)` / `SolutionLoop.from_config(...)` take the
`TimeIntegration` plugin in place of the `AdvanceMode` argument.

## Step-by-step plan

1. **New file `src/neofoam/framework/time_integration.py`** — the
   `TimeIntegration` family + `TransientIntegration` / `SteadyIntegration` +
   `integration_from_ddt`. (`TimeControlConfig` referenced via `TYPE_CHECKING` to
   avoid an import cycle with `foam_time.py`.)
2. **`foam_time.py`** — replace the `AdvanceMode` field with a `TimeIntegration`;
   route `initial_delta_t` / `timeName` through it; delete `AdvanceMode`,
   `advance_mode_from_ddt`, and `delta_t_constraints_from`. Keep
   `TimeControlConfig` where it is (or move it beside the family — see Risks).
3. **`solution_loop.py`** — `from_config` accepts `integration` instead of
   `mode`; seed constraints from `integration.constraints(config)` plus the
   transient `CourantConstraint`.
4. **`write_control.py`** — rewrite `write_control_from_config` to build via
   `WriteControl.create(policy=payload)`; add `_policy_payload` (or a
   `WriteControlConfig.to_policy_payload`).
5. **Solver Models** — update `models/solution_loop.py` build to select the
   integration; `models/field_writer.py` needs no change.
6. **Tests** — see below.
7. **(Follow-up)** elevate `control.py` to a `LoopControl` family on `BaseConfig`.

## Test impact

Mirrors `src/` 1:1 (one `test_<name>.py` per source file; free functions). Run
with the cross-venv command from the `enh+pytimeloop` worktree memory.

- **New** `test/framework/test_time_integration.py` — `initial_delta_t`,
  `step_name` (float `%g` vs. integer iteration index), `constraints` per
  variant; round-trips `TimeIntegration.create(...)` from a payload.
- `test/framework/test_foam_time.py` / `test_foam_time_parity.py` — swap
  `mode=AdvanceMode.ITERATION` for the steady integration; parity assertions
  (`operator++`, `deltaT0`, `timeName`) must stay byte-identical.
- `test/framework/test_advance_mode.py` — rename/retarget to the integration
  selector (`integration_from_ddt("steadyState")` → steady).
- `test/framework/test_write_control.py` — add a case proving a freshly
  `@WriteControl.register`-ed policy is selectable purely by keyword (no factory
  edit), i.e. selection goes through the union.
- `test/framework/test_solution_loop.py`,
  `test/solver/incompressibleFluid/test_solution_loop.py` — update construction;
  steady run still ends correctly, transient still runs to `endTime`.

## Risks / notes

- **Import cycle.** `TimeIntegration.constraints` needs `TimeControlConfig`, and
  `foam_time.py` would import `TimeIntegration`. Resolve by keeping
  `TimeControlConfig` in `foam_time.py` and importing it under `TYPE_CHECKING`
  inside `time_integration.py` (the methods only need it at call time), or by
  moving `TimeControlConfig` into a small shared config module both import.
- **Behavioural parity is the acceptance bar.** `timeName`, `operator++`,
  `deltaT0` bookkeeping and the adjustable-write rounding must not change — the
  parity tests are the guard. This refactor is a *structure* change, not a
  numerics change.
- **`stepper_decides=True`** (the default in `field_writer.py`) keeps using
  `StepperWriteControl` — the union path only matters when the explicit policies
  drive the decision; both paths stay supported.
- **`.create(...)` ergonomics.** `PluginSystem` builds a wrapper model with the
  discriminated field (`integration` / `policy`); unwrap with `.integration` /
  `.policy`. Confirm against `write_control.py`'s existing usage so the pattern
  matches the codebase.

## Definition of done

- No `AdvanceMode` enum and no `mode is AdvanceMode.*` branch remains.
- Write-control policy selection resolves through the `WriteControl`
  discriminated union; adding a policy needs no edit to `write_control_from_config`.
- `TimeIntegration` is a registered `PluginSystem` family parameterised by
  `BaseConfig`, selected from the case (`ddtSchemes`).
- All existing parity/loop tests pass unchanged in behaviour; new family tests
  added.
