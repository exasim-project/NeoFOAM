# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the SolutionLoop engine *and* its solutionLoop core Model.

Engine — the pure-Python main iteration loop unifying steady and unsteady::

    while loop.running():        # SolutionControl: advance / stop on convergence
        loop.set_courant(Co)     # a flow model pushes the measured Courant number
        loop.adjust_delta_t()    # min over injected DeltaTConstraints (CFL, maxDeltaT)
        loop.advance()           # stepper.increment()
        ...solve...

Model — the backend-agnostic wrapper: builds the stepper + engine, exposes the
loop body as operations, and injects the Courant provider / logger / StepSink via
``ctx.models`` seams (no-op defaults). The step update is pure Python (FoamTime);
the stepper only *pushes* each step to a StepSink. No OpenFOAM needed.
"""

from __future__ import annotations

from typing import Any, Optional, cast

from neofoam.algorithms.control import SolutionControl
from neofoam.algorithms.foam_time import FoamTime, TimeControlConfig
from neofoam.algorithms.solution_loop import (
    SolutionLoop,
    SolutionLoopPredicate,
    build,
    increment_time,
    make_solution_loop,
    make_stepper,
    set_time_step,
    solutionLoop,
)
from neofoam.algorithms.time_integration import TimeIntegration, TransientIntegration
from neofoam.algorithms.time_step import CourantConstraint, MaxDeltaTConstraint
from neofoam.framework.context import Context


class FakeSink:
    """Records the step updates the stepper pushes (stands in for a backend)."""

    def __init__(self) -> None:
        self.deltas: list[float] = []
        self.advances: list[tuple[float, int]] = []

    def set_delta_t(self, dt: float) -> None:
        self.deltas.append(dt)

    def advance_to(self, value: float, index: int) -> None:
        self.advances.append((value, index))


def _stepper(
    *,
    end: float = 0.3,
    dt: float = 0.1,
    write_interval: float = 1.0,
    write_control: str = "timeStep",
    sink: Optional[FakeSink] = None,
    integration: Optional[TimeIntegration] = None,
) -> FoamTime:
    return FoamTime(
        end_time=end,
        delta_t=dt,
        write_interval=write_interval,
        write_control=write_control,
        integration=integration if integration is not None else TransientIntegration(),
        sink=sink,
    )


def _config(**kw: object) -> TimeControlConfig:
    base: dict[str, object] = {"endTime": 0.3, "deltaT": 0.1}
    base.update(kw)
    return TimeControlConfig(**base)


class FakeContext:
    """Stands in for Context; the loop ops only read ``.models``."""

    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models


# =========================================================================
# Engine
# =========================================================================


# --- transient: empty residualControl -> runs to endTime -----------------


def test_transient_runs_to_end_time() -> None:
    loop = SolutionLoop(stepper=_stepper(end=0.3, dt=0.1), control=SolutionControl())
    steps = 0
    while loop.running():
        loop.advance()
        steps += 1
    assert steps == 3
    assert abs(loop.stepper.value() - 0.3) < 1e-12


def test_running_delegates_to_solution_control() -> None:
    control = SolutionControl()
    control.store_residual("p", 0.0)
    loop = SolutionLoop(stepper=_stepper(end=1.0, dt=0.1), control=control)
    assert loop.running() is True


# --- steady: residualControl set -> stops on convergence -----------------


def test_steady_stops_on_convergence() -> None:
    control = SolutionControl(residualControl={"p": 1e-2, "U": 1e-3})
    loop = SolutionLoop(stepper=_stepper(end=100.0, dt=1.0), control=control)
    assert loop.running() is True
    control.store_residual("p", 1e-3)
    control.store_residual("U", 1e-4)
    assert loop.running() is False  # converged -> stepper stopped


def test_convergence_stop_does_not_write() -> None:
    # ending the run on convergence must not flip the stepper into a write step
    control = SolutionControl(residualControl={"p": 1e-2})
    loop = SolutionLoop(stepper=_stepper(end=100.0, dt=1.0), control=control)
    control.store_residual("p", 1e-3)  # converged
    assert loop.running() is False
    assert loop.stepper.outputTime() is False


# --- injectable stability criteria (Courant pushed in) -------------------


def test_adjust_delta_t_no_constraints_keeps_step() -> None:
    loop = SolutionLoop(stepper=_stepper(dt=0.1))
    loop.adjust_delta_t()
    assert loop.stepper.deltaTValue() == 0.1


def test_courant_constraint_shrinks_step() -> None:
    loop = SolutionLoop(
        stepper=_stepper(dt=0.1),
        constraints=[CourantConstraint(maxCo=1.0)],
    )
    loop.set_courant(2.0)  # too fast
    loop.adjust_delta_t()
    # dt * maxCo / Co = 0.1 * 1.0 / 2.0 = 0.05
    assert loop.stepper.deltaTValue() == 0.05


def test_growth_cap_limits_increase() -> None:
    loop = SolutionLoop(
        stepper=_stepper(dt=0.1),
        constraints=[CourantConstraint(maxCo=1.0)],
        growth_cap=1.2,
    )
    loop.set_courant(0.1)  # plenty of headroom -> would grow 10x
    loop.adjust_delta_t()
    assert abs(loop.stepper.deltaTValue() - 0.12) < 1e-12


def test_min_over_multiple_constraints() -> None:
    loop = SolutionLoop(
        stepper=_stepper(dt=0.1),
        constraints=[
            CourantConstraint(maxCo=1.0),
            MaxDeltaTConstraint(maxDeltaT=0.11),  # the binding cap
        ],
        growth_cap=10.0,
    )
    loop.set_courant(0.1)  # Courant allows large dt
    loop.adjust_delta_t()
    assert abs(loop.stepper.deltaTValue() - 0.11) < 1e-12


def test_add_constraint_injection() -> None:
    loop = SolutionLoop(stepper=_stepper(dt=0.1))
    loop.set_courant(2.0)
    loop.adjust_delta_t()
    assert loop.stepper.deltaTValue() == 0.1  # no constraint yet
    loop.add_constraint(CourantConstraint(maxCo=1.0))
    loop.adjust_delta_t()
    assert loop.stepper.deltaTValue() == 0.05  # constraint now applies


# --- (t, dt, dt0) surfaced through the stepper ----------------------------


def test_stepper_exposes_t_dt_dt0() -> None:
    loop = SolutionLoop(stepper=_stepper(end=0.3, dt=0.1))
    loop.advance()
    loop.advance()
    stepper = loop.stepper
    assert abs(stepper.value() - 0.2) < 1e-12
    assert stepper.deltaTValue() == 0.1
    assert stepper.deltaT0Value() == 0.1  # previous step size


# --- the stepper pushes each step to the StepSink -------------------------


def test_stepper_pushes_steps_to_sink() -> None:
    sink = FakeSink()
    loop = SolutionLoop(
        stepper=_stepper(end=0.3, dt=0.1, sink=sink),
        constraints=[CourantConstraint(maxCo=1.0)],
    )
    loop.set_courant(0.5)
    while loop.running():
        loop.adjust_delta_t()  # -> sink.set_delta_t
        loop.advance()  # -> sink.advance_to
    assert len(sink.deltas) == len(sink.advances) >= 1
    indices = [idx for _, idx in sink.advances]
    assert indices == list(range(1, len(indices) + 1))  # sequential step indices


def test_null_sink_is_the_default() -> None:
    # no sink supplied -> advancement still works, nothing to push to
    loop = SolutionLoop(stepper=_stepper(end=0.2, dt=0.1))
    steps = 0
    while loop.running():
        loop.advance()
        steps += 1
    assert steps == 2


# =========================================================================
# Model (backend-agnostic core Model)
# =========================================================================


def test_modelspec_is_a_full_core_model() -> None:
    assert solutionLoop.name == "solutionLoop"
    assert solutionLoop._load_func is not None
    assert solutionLoop._build_func is not None
    op_names = {meta["name"] for _, meta in solutionLoop._operations}
    assert op_names == {"set_time_step", "increment_time"}


def test_build_emits_stepper_then_engine() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=0.5)
    steps = build(config)
    assert [s.name for s in steps] == ["models.stepper", "models.solution_loop"]

    stepper = steps[0].initializer({})
    assert isinstance(stepper, FoamTime)

    assert "models.stepper" in steps[1].depends_on
    loop = steps[1].initializer({"models.stepper": stepper})
    assert isinstance(loop, SolutionLoop)
    assert loop.stepper is stepper
    assert {type(c).__name__ for c in loop.constraints} == {
        "CourantConstraint",
        "MaxDeltaTConstraint",
    }


# --- constraint seeding from config (make_solution_loop) ------------------


def test_adjustable_config_injects_courant_and_maxdeltat() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=0.5)
    loop = make_solution_loop(config, make_stepper(config))
    assert {type(c).__name__ for c in loop.constraints} == {
        "CourantConstraint",
        "MaxDeltaTConstraint",
    }


def test_fixed_step_config_injects_no_constraints() -> None:
    loop = make_solution_loop(_config(adjustTimeStep=False), make_stepper(_config()))
    assert loop.constraints == []


# --- injected Courant provider seam ---------------------------------------


def test_set_time_step_pushes_courant_from_injected_provider() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=10.0)
    loop = make_solution_loop(config, make_stepper(config))
    seen: list[Any] = []

    def provider(ctx: Any) -> float:
        seen.append(ctx)
        return 2.0

    ctx = cast(
        Context,
        FakeContext({"solution_loop": loop, "courant_provider": provider}),
    )
    set_time_step(None, ctx)
    assert seen  # provider was consulted
    assert loop.stepper.deltaTValue() == 0.05  # 0.1 * 1.0 / 2.0 (CFL-limited)


def test_set_time_step_without_provider_just_adjusts() -> None:
    # no courant_provider injected: fixed step is left unchanged, no crash
    loop = make_solution_loop(_config(adjustTimeStep=False), make_stepper(_config()))
    ctx = cast(Context, FakeContext({"solution_loop": loop}))
    set_time_step(None, ctx)
    assert loop.stepper.deltaTValue() == 0.1


# --- injected logger seam -------------------------------------------------


def test_increment_time_uses_injected_logger_and_advances() -> None:
    loop = make_solution_loop(_config(), make_stepper(_config()))
    logs: list[str] = []
    ctx = cast(
        Context, FakeContext({"solution_loop": loop, "loop_logger": logs.append})
    )
    increment_time(None, ctx)
    assert loop.stepper.timeIndex() == 1
    # logged before advance (faithful to the existing solver op): the step name
    # is still the pre-advance value
    assert logs == ["Time = 0"]


# --- predicate ------------------------------------------------------------


def test_predicate_delegates_to_engine_running() -> None:
    on = make_solution_loop(_config(endTime=1.0), make_stepper(_config(endTime=1.0)))
    off = make_solution_loop(_config(endTime=1.0), make_stepper(_config(endTime=1.0)))
    off.stepper.stop()
    assert (
        SolutionLoopPredicate()(cast(Context, FakeContext({"solution_loop": on})))
        is True
    )
    assert (
        SolutionLoopPredicate()(cast(Context, FakeContext({"solution_loop": off})))
        is False
    )
