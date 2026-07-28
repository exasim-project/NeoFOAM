# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the SolutionLoop engine *and* its solutionLoop core Model.

Engine — the pure-Python main iteration loop unifying steady and unsteady; it
advances a :class:`LoopState` and mirrors it onto an optional :class:`LoopBackend`::

    while loop.running():                       # SolutionControl: advance / stop
        loop.constrain_delta_t(timeStepConstraint())  # folded model limits
        loop.advance()                          # ++ the LoopState
        ...solve...

Model — the backend-agnostic wrapper: builds the LoopState (ctx.time) + a *bare*
engine, exposes the loop body as operations, and folds the model-owned
``timeStepConstraint``/``loopCondition`` interfaces. No OpenFOAM needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.control import SolutionControl
from neofoam.algorithms.solution_loop.interfaces import (
    VGREAT,
    initialTimeStepConstraint,
    loopCondition,
    maxTimeStep,
    timeStepConstraint,
)
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.solution_loop import (
    SolutionLoop,
    SolutionLoopPredicate,
    build,
    increment_time,
    make_loop_state,
    make_solution_loop,
    set_time_step,
    solutionLoop,
)
from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.model import Model, ModelRuntime, bind_owned_interfaces

# Advancement is exact float arithmetic (t += dt), so the only error is dt summed
# over a handful of steps — a few ULP. 1e-12 sits comfortably above that and well
# below any physically meaningful time difference for these sub-second cases.
STEP_ATOL = 1e-12


class FakeBackend:
    """Records the LoopState snapshots the engine mirrors (a fake LoopBackend)."""

    def __init__(self) -> None:
        self.updates: list[tuple[float, float, int]] = []

    def update(self, state: LoopState) -> None:
        self.updates.append((state.delta_t, state.value, state.index))


def _state(
    *,
    end: float = 0.3,
    dt: float = 0.1,
    write_interval: float = 1.0,
    write_control: str = "timeStep",
) -> LoopState:
    return LoopState(
        value=0.0,
        delta_t=dt,
        end_time=end,
        write_control=write_control,
        write_interval=write_interval,
    )


def _config(**kw: object) -> TimeControlConfig:
    base: dict[str, object] = {"endTime": 0.3, "deltaT": 0.1}
    base.update(kw)
    return TimeControlConfig(**base)


def _ctx(models: dict[str, Any]) -> Context:
    """A real Context around ``models`` — the loop ops/predicate read only
    ``.models``, but build the genuine object rather than a duck-typed shim (a
    real Context is a one-liner, as ``_drive_set_time_step`` also shows)."""
    return Context(fields={}, models=models)


# =========================================================================
# Engine
# =========================================================================


def test_transient_runs_to_end_time() -> None:
    loop = SolutionLoop(state=_state(end=0.3, dt=0.1), control=SolutionControl())
    steps = 0
    while loop.running():
        loop.advance()
        steps += 1
    assert steps == 3
    assert abs(loop.state.value - 0.3) < STEP_ATOL


def test_running_delegates_to_solution_control() -> None:
    control = SolutionControl()
    control.store_residual("p", 0.0)
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1), control=control)
    assert loop.running() is True


def test_steady_stops_on_convergence() -> None:
    control = SolutionControl(residualControl={"p": 1e-2, "U": 1e-3})
    loop = SolutionLoop(state=_state(end=100.0, dt=1.0), control=control)
    assert loop.running() is True
    control.store_residual("p", 1e-3)
    control.store_residual("U", 1e-4)
    assert loop.running() is False  # converged -> run ended


def test_convergence_stop_does_not_write() -> None:
    control = SolutionControl(residualControl={"p": 1e-2})
    loop = SolutionLoop(state=_state(end=100.0, dt=1.0), control=control)
    control.store_residual("p", 1e-3)  # converged
    assert loop.running() is False
    assert loop.state.write_time is False


# --- next deltaT folds the timeStepConstraint limit (single path) ---------


def test_constrain_delta_t_no_opinion_keeps_step() -> None:
    loop = SolutionLoop(state=_state(dt=0.1))
    loop.constrain_delta_t(VGREAT)  # min([], default=VGREAT) -> no active opinion
    assert loop.state.delta_t == 0.1


def test_constrain_delta_t_applies_a_binding_limit() -> None:
    loop = SolutionLoop(state=_state(dt=0.1), growth_cap=10.0)
    loop.constrain_delta_t(0.05)  # below growth cap -> applied directly
    assert loop.state.delta_t == pytest.approx(0.05)


def test_constrain_delta_t_clamps_growth_to_the_cap() -> None:
    # A huge folded limit must be clamped to growth_cap * current, not applied raw.
    loop = SolutionLoop(state=_state(dt=0.1), growth_cap=1.2)
    loop.constrain_delta_t(10.0)
    assert loop.state.delta_t == pytest.approx(0.12)  # 1.2 * 0.1, not 10.0


def test_constrain_delta_t_clips_the_ceiling_after_the_damping() -> None:
    # setDeltaT.H: min(deltaTFact*deltaT, maxDeltaT). Damping the ceiling instead
    # of clipping with it would give min(1.15, 1.115, 1.2)*0.1 = 0.1115.
    loop = SolutionLoop(state=_state(dt=0.1), growth_cap=1.2)
    loop.constrain_delta_t(0.12, ceiling=0.115)
    assert loop.state.delta_t == pytest.approx(0.112)  # min(1.12, 1.2) * 0.1


def test_constrain_delta_t_with_no_opinion_does_not_snap() -> None:
    # adjustTimeStep off: OpenFOAM never reaches Time::setDeltaT, so the
    # adjustableRunTime snapping must not run either (0.35/round(3.5) = 0.0875).
    loop = SolutionLoop(
        state=_state(dt=0.1, write_control="adjustableRunTime", write_interval=0.35)
    )
    loop.constrain_delta_t(VGREAT)
    assert loop.state.delta_t == 0.1


def test_initial_delta_t_reduces_undamped() -> None:
    # setInitialDeltaT.H applies the CFL limit in full, ignoring the growth damping.
    loop = SolutionLoop(state=_state(dt=0.1), growth_cap=1.2)
    loop.set_initial_delta_t(0.02)
    assert loop.state.delta_t == pytest.approx(0.02)


def test_initial_delta_t_never_raises_the_step() -> None:
    loop = SolutionLoop(state=_state(dt=0.1))
    loop.set_initial_delta_t(10.0)
    assert loop.state.delta_t == 0.1


def test_initial_delta_t_skipped_on_a_quiescent_flow() -> None:
    # No opinion (Co <= SMALL) -> the whole pass, snapping included, is skipped.
    loop = SolutionLoop(
        state=_state(dt=0.1, write_control="adjustableRunTime", write_interval=0.35)
    )
    loop.set_initial_delta_t(VGREAT)
    assert loop.state.delta_t == 0.1


# --- (t, dt, dt0) on the LoopState ----------------------------------------


def test_state_exposes_t_dt_dt0() -> None:
    loop = SolutionLoop(state=_state(end=0.3, dt=0.1))
    loop.advance()
    loop.advance()
    s = loop.state
    assert abs(s.value - 0.2) < STEP_ATOL
    assert s.delta_t == 0.1
    assert s.delta_t0 == 0.1  # previous step size


# --- the engine mirrors each step onto the LoopBackend --------------------


def test_engine_mirrors_state_to_backend() -> None:
    backend = FakeBackend()
    loop = SolutionLoop(state=_state(end=0.3, dt=0.1), backend=backend)
    while loop.running():
        loop.constrain_delta_t(VGREAT)  # -> backend.update
        loop.advance()  # -> backend.update
    indices = [idx for _dt, _v, idx in backend.updates]
    assert indices == sorted(indices)
    assert indices[-1] == loop.state.index >= 1


def test_set_backend_injects_and_syncs() -> None:
    backend = FakeBackend()
    loop = SolutionLoop(state=_state(dt=0.1))
    loop.set_backend(backend)  # injected post-construction -> immediate sync
    assert backend.updates  # synced the initial state on injection
    loop.advance()
    assert backend.updates[-1][2] == 1  # latest update reflects the advanced index


def test_null_backend_is_the_default() -> None:
    loop = SolutionLoop(state=_state(end=0.2, dt=0.1))
    steps = 0
    while loop.running():
        loop.advance()
        steps += 1
    assert steps == 2


# --- the redundant measurement/constraint seam is gone --------------------


def test_engine_has_no_legacy_constraint_or_measurement_members() -> None:
    loop = SolutionLoop(state=_state(dt=0.1))
    for attr in (
        "publish",
        "measured",
        "add_constraint",
        "adjust_delta_t",
        "constraints",
    ):
        assert not hasattr(loop, attr)


def test_constraints_time_step_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("neofoam.algorithms.constraints.time_step")


# =========================================================================
# Model (backend-agnostic core Model)
# =========================================================================


def test_modelspec_is_a_full_core_model() -> None:
    assert solutionLoop.name == "solutionLoop"
    assert solutionLoop._load_func is not None
    assert solutionLoop._build_func is not None
    op_names = {meta["name"] for _, meta in solutionLoop._operations}
    assert op_names == {"set_time_step", "increment_time"}


def test_build_emits_state_then_engine() -> None:
    config = _config()
    steps = build(config)
    assert [s.name for s in steps] == ["time", "models.solution_loop"]

    state = steps[0].initializer({})
    assert isinstance(state, LoopState)

    assert "time" in steps[1].depends_on
    loop = steps[1].initializer({"time": state})
    assert isinstance(loop, SolutionLoop)
    assert loop.state is state


def test_increment_time_uses_injected_logger_and_advances() -> None:
    loop = make_solution_loop(_config(), make_loop_state(_config()))
    logs: list[str] = []
    ctx = _ctx({"solution_loop": loop, "loop_logger": logs.append})
    increment_time(None, ctx)
    assert loop.state.index == 1
    assert logs == ["Time = 0"]


def test_predicate_delegates_to_engine_running() -> None:
    on = make_solution_loop(_config(endTime=1.0), make_loop_state(_config(endTime=1.0)))
    off = make_solution_loop(_config(endTime=1.0), make_loop_state(_config(endTime=1.0)))
    off.stop()
    assert SolutionLoopPredicate()(_ctx({"solution_loop": on})) is True
    assert SolutionLoopPredicate()(_ctx({"solution_loop": off})) is False


@pytest.mark.parametrize(
    "conds, expected",
    [
        ([lambda lp: False], False),
        ([], True),
        ([lambda lp: True, lambda lp: lp.current_delta_t() > 0.0], True),
    ],
)
def test_constructor_folds_conditions_with_all(conds: list[Any], expected: bool) -> None:
    loop = SolutionLoop(state=_state(), conditions=conds)
    assert loop.all_conditions_hold() is expected
    assert loop.conditions == loop._conditions


# --- the loop model owns both interfaces ----------------------------------


def test_loop_model_owns_its_interfaces() -> None:
    assert solutionLoop.declared_interfaces["timeStepConstraint"] is timeStepConstraint
    assert solutionLoop.declared_interfaces["maxTimeStep"] is maxTimeStep
    assert (
        solutionLoop.declared_interfaces["initialTimeStepConstraint"] is initialTimeStepConstraint
    )
    assert solutionLoop.declared_interfaces["loopCondition"] is loopCondition


def test_loop_module_keeps_live_interface_annotations() -> None:
    ann = set_time_step.__annotations__
    assert ann["constraints"] is timeStepConstraint
    assert ann["ceilings"] is maxTimeStep
    assert ann["initial_constraints"] is initialTimeStepConstraint
    assert ann["conditions"] is loopCondition


# --- the interface-consuming loop-body operation --------------------------


def _drive_set_time_step(loop: SolutionLoop, contributors: list[ModelRuntime]) -> Context:
    loop_rt = ModelRuntime(spec=solutionLoop, name="solutionLoop", config=None)
    models: dict[str, Any] = {"solution_loop": loop, "solutionLoop": loop_rt}
    for rt in contributors:
        models[rt.name] = rt
    ctx = Context(fields={}, models=models)
    bind_owned_interfaces(loop_rt, contributors, ctx)
    wrap_with_dependency_resolution(
        set_time_step, instance=None, dependency_resolver=DependencyResolver()
    )(ctx)
    return ctx


def test_set_time_step_is_fixed_step_with_no_contributors() -> None:
    loop = SolutionLoop(state=_state(dt=0.1))
    _drive_set_time_step(loop, [])
    assert loop.state.delta_t == pytest.approx(0.1)  # min([], default=VGREAT) -> kept


def test_set_time_step_records_keep_running_true_with_no_conditions() -> None:
    loop = SolutionLoop(state=_state(dt=0.1))
    loop.keep_running = False  # sentinel
    _drive_set_time_step(loop, [])
    assert loop.keep_running is True


def test_set_time_step_publishes_current_step_as_injectable() -> None:
    loop = SolutionLoop(state=_state(dt=0.2))
    ctx = _drive_set_time_step(loop, [])
    assert ctx.fields["deltaT"] == pytest.approx(0.2)


def test_predicate_runs_when_keep_running_is_true() -> None:
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1))
    ctx = _ctx({"solution_loop": loop})
    assert SolutionLoopPredicate()(ctx) is True


def test_predicate_stops_when_keep_running_is_false() -> None:
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1))
    loop.keep_running = False
    ctx = _ctx({"solution_loop": loop})
    assert SolutionLoopPredicate()(ctx) is False


_loop_stopper = Model("loopStopper")


@_loop_stopper.contributes(loopCondition)
def _vote_stop() -> bool:
    return False


def _stopper_runtime() -> ModelRuntime:
    return ModelRuntime(spec=_loop_stopper, name="loopStopper", config=None)


def test_predicate_runs_when_no_condition_vetoes() -> None:
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1))
    ctx = _drive_set_time_step(loop, [])
    assert loop.keep_running is True
    assert SolutionLoopPredicate()(ctx) is True


def test_predicate_stops_when_a_condition_vetoes() -> None:
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1))
    ctx = _drive_set_time_step(loop, [_stopper_runtime()])
    assert loop.keep_running is False
    assert SolutionLoopPredicate()(ctx) is False


# --- min over active contributions, then growth-clamped -------------------

_cap_low = Model("capLow")
_cap_high = Model("capHigh")


@_cap_low.contributes(timeStepConstraint)
def _limit_low() -> float:
    return 0.2


@_cap_high.contributes(timeStepConstraint)
def _limit_high() -> float:
    return 0.5


def test_fold_takes_min_then_growth_clamp() -> None:
    # {0.2, 0.5} at dt=0.1, growth 1.2 -> min(0.2, 0.5, 1.2*0.1) = 0.12.
    loop = SolutionLoop(state=_state(dt=0.1), growth_cap=1.2)
    contribs = [
        ModelRuntime(spec=_cap_low, name="capLow", config=None),
        ModelRuntime(spec=_cap_high, name="capHigh", config=None),
    ]
    _drive_set_time_step(loop, contribs)
    assert loop.next_dt == pytest.approx(0.2)  # the folded min limit
    assert loop.state.delta_t == pytest.approx(0.12)  # growth clamp binds


# --- the published current step flows into a deltaT-taking contribution ----

_half_stepper = Model("halfStepper")


@_half_stepper.contributes(timeStepConstraint)
def _half_of_step(deltaT: float) -> float:
    return deltaT * 0.5


def _half_runtime() -> ModelRuntime:
    return ModelRuntime(spec=_half_stepper, name="halfStepper", config=None)


def test_published_step_flows_into_a_delta_t_contribution() -> None:
    loop = SolutionLoop(state=_state(dt=0.2))
    _drive_set_time_step(loop, [_half_runtime()])
    assert loop.state.delta_t == pytest.approx(0.1)  # 0.5 * published 0.2
