# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the SolutionLoop engine *and* its solutionLoop core Model.

Engine — the pure-Python main iteration loop unifying steady and unsteady; it
advances a :class:`LoopState` and mirrors it onto an optional :class:`LoopBackend`::

    while loop.running():        # SolutionControl: advance / stop on convergence
        loop.set_courant(Co)     # a flow model pushes the measured Courant number
        loop.adjust_delta_t()    # min over injected DeltaTConstraints (CFL, maxDeltaT)
        loop.advance()           # ++ the LoopState
        ...solve...

Model — the backend-agnostic wrapper: builds the LoopState (ctx.time) + engine,
exposes the loop body as operations, and injects the Courant provider / logger /
LoopBackend via ``ctx.models`` seams (no-op defaults). No OpenFOAM needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import neofoam.algorithms.solution_loop as loop_pkg
import neofoam.algorithms.solution_loop.solution_loop as engine_mod
from neofoam.algorithms.constraints.time_step import (
    CourantConstraint,
    MaxDeltaTConstraint,
)
from neofoam.algorithms.solution_loop.conditions import ConditionVote
from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.control import SolutionControl
from neofoam.algorithms.solution_loop.interfaces import (
    loopCondition,
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


class FakeContext:
    """Stands in for Context; the loop ops only read ``.models``."""

    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models


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
    assert abs(loop.state.value - 0.3) < 1e-12


# --- injectable stability criteria (Courant pushed in) -------------------


def test_adjust_delta_t_no_constraints_keeps_step() -> None:
    loop = SolutionLoop(state=_state(dt=0.1))
    loop.adjust_delta_t()
    assert loop.state.delta_t == 0.1


def test_courant_constraint_shrinks_step() -> None:
    loop = SolutionLoop(
        state=_state(dt=0.1), constraints=[CourantConstraint(maxCo=1.0)]
    )
    loop.set_courant(2.0)  # too fast
    loop.adjust_delta_t()
    assert loop.state.delta_t == 0.05  # 0.1 * 1.0 / 2.0


def test_growth_cap_limits_increase() -> None:
    loop = SolutionLoop(
        state=_state(dt=0.1),
        constraints=[CourantConstraint(maxCo=1.0)],
        growth_cap=1.2,
    )
    loop.set_courant(0.1)  # plenty of headroom -> would grow 10x
    loop.adjust_delta_t()
    assert abs(loop.state.delta_t - 0.12) < 1e-12


def test_min_over_multiple_constraints() -> None:
    loop = SolutionLoop(
        state=_state(dt=0.1),
        constraints=[
            CourantConstraint(maxCo=1.0),
            MaxDeltaTConstraint(maxDeltaT=0.11),  # the binding cap
        ],
        growth_cap=10.0,
    )
    loop.set_courant(0.1)
    loop.adjust_delta_t()
    assert abs(loop.state.delta_t - 0.11) < 1e-12


def test_add_constraint_injection() -> None:
    loop = SolutionLoop(state=_state(dt=0.1))
    loop.set_courant(2.0)
    loop.adjust_delta_t()
    assert loop.state.delta_t == 0.1  # no constraint yet
    loop.add_constraint(CourantConstraint(maxCo=1.0))
    loop.adjust_delta_t()
    assert loop.state.delta_t == 0.05  # constraint now applies


# --- (t, dt, dt0) on the LoopState ----------------------------------------


def test_state_exposes_t_dt_dt0() -> None:
    loop = SolutionLoop(state=_state(end=0.3, dt=0.1))
    loop.advance()
    loop.advance()
    s = loop.state
    assert abs(s.value - 0.2) < 1e-12
    assert s.delta_t == 0.1
    assert s.delta_t0 == 0.1  # previous step size


# --- the engine mirrors each step onto the LoopBackend --------------------


def test_engine_mirrors_state_to_backend() -> None:
    backend = FakeBackend()
    loop = SolutionLoop(
        state=_state(end=0.3, dt=0.1),
        constraints=[CourantConstraint(maxCo=1.0)],
        backend=backend,
    )
    loop.set_courant(0.5)
    while loop.running():
        loop.adjust_delta_t()  # -> backend.update
        loop.advance()  # -> backend.update
    # the backend was mirrored every step; indices are monotonic up to the last
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
    # no backend supplied -> advancement still works, nothing to mirror to
    loop = SolutionLoop(state=_state(end=0.2, dt=0.1))
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


def test_build_emits_state_then_engine() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=0.5)
    steps = build(config)
    assert [s.name for s in steps] == [
        "time",
        "models.solution_loop",
    ]

    state = steps[0].initializer({})
    assert isinstance(state, LoopState)

    assert "time" in steps[1].depends_on
    loop = steps[1].initializer({"time": state})
    assert isinstance(loop, SolutionLoop)
    assert loop.state is state


# --- constraint seeding from config (make_solution_loop) ------------------


def test_make_solution_loop_seeds_no_constraints() -> None:
    # Stability limits are folded per step from the timeStepConstraint interface,
    # not seeded here — so the engine starts with an empty constraint list for any
    # config (adjustable or fixed).
    adjustable = make_solution_loop(
        _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=0.5),
        make_loop_state(_config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=0.5)),
    )
    fixed = make_solution_loop(
        _config(adjustTimeStep=False), make_loop_state(_config())
    )
    assert adjustable.constraints == []
    assert fixed.constraints == []


# --- injected logger seam -------------------------------------------------


def test_increment_time_uses_injected_logger_and_advances() -> None:
    loop = make_solution_loop(_config(), make_loop_state(_config()))
    logs: list[str] = []
    ctx = cast(
        Context, FakeContext({"solution_loop": loop, "loop_logger": logs.append})
    )
    increment_time(None, ctx)
    assert loop.state.index == 1
    # logged before advance (faithful to the solver op): pre-advance step name
    assert logs == ["Time = 0"]


# --- predicate ------------------------------------------------------------


def test_predicate_delegates_to_engine_running() -> None:
    on = make_solution_loop(_config(endTime=1.0), make_loop_state(_config(endTime=1.0)))
    off = make_solution_loop(
        _config(endTime=1.0), make_loop_state(_config(endTime=1.0))
    )
    off.stop()
    assert (
        SolutionLoopPredicate()(cast(Context, FakeContext({"solution_loop": on})))
        is True
    )
    assert (
        SolutionLoopPredicate()(cast(Context, FakeContext({"solution_loop": off})))
        is False
    )


# --- constructor DI for loop conditions (folded with `all`) ---------------


@pytest.mark.parametrize(
    "conds, expected",
    [
        ([lambda lp: False], False),
        ([], True),
        ([lambda lp: True, lambda lp: lp.current_delta_t() > 0.0], True),
    ],
)
def test_constructor_folds_conditions_with_all(
    conds: list[Any], expected: bool
) -> None:
    loop = SolutionLoop(state=_state(), conditions=conds)
    assert loop.all_conditions_hold() is expected
    assert loop.conditions == loop._conditions  # public getter mirrors the backing list


def test_constructor_still_accepts_constraints() -> None:
    loop = SolutionLoop(
        state=_state(), constraints=[MaxDeltaTConstraint(maxDeltaT=0.5)]
    )
    assert [type(c).__name__ for c in loop.constraints] == ["MaxDeltaTConstraint"]


# --- the loop model owns both interfaces ----------------------------------


def test_loop_model_owns_both_interfaces() -> None:
    assert solutionLoop.declared_interfaces["timeStepConstraint"] is timeStepConstraint
    assert solutionLoop.declared_interfaces["loopCondition"] is loopCondition


# --- the interface-consuming loop-body operation --------------------------


def test_loop_module_keeps_live_interface_annotations() -> None:
    # Proves the module did NOT stringify annotations: the param annotations are
    # the live spec objects, not strings.
    ann = set_time_step.__annotations__
    assert ann["constraints"] is timeStepConstraint
    assert ann["conditions"] is loopCondition


def _drive_set_time_step(
    loop: SolutionLoop, contributors: list[ModelRuntime]
) -> Context:
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


# --- predicate ANDs the folded loopCondition (keep_running) ---------------


def test_predicate_runs_when_keep_running_is_true() -> None:
    loop = SolutionLoop(
        state=_state(end=1.0, dt=0.1)
    )  # mid-run, keep_running default True
    ctx = cast(Context, FakeContext({"solution_loop": loop}))
    assert SolutionLoopPredicate()(ctx) is True


def test_predicate_stops_when_keep_running_is_false() -> None:
    loop = SolutionLoop(
        state=_state(end=1.0, dt=0.1)
    )  # would otherwise still be running
    loop.keep_running = False
    ctx = cast(Context, FakeContext({"solution_loop": loop}))
    assert SolutionLoopPredicate()(ctx) is False


# --- the folded loopCondition reaches the predicate via set_time_step -----

_loop_stopper = Model("loopStopper")


@_loop_stopper.contributes(loopCondition)
def _vote_stop() -> ConditionVote:
    return ConditionVote(satisfied=True)


_loop_aborter = Model("loopAborter")


@_loop_aborter.contributes(loopCondition)
def _vote_abort() -> ConditionVote:
    return ConditionVote(satisfied=True, action="abort")


def _stopper_runtime() -> ModelRuntime:
    return ModelRuntime(spec=_loop_stopper, name="loopStopper", config=None)


def _aborter_runtime() -> ModelRuntime:
    return ModelRuntime(spec=_loop_aborter, name="loopAborter", config=None)


def test_predicate_runs_when_no_condition_is_satisfied() -> None:
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1))
    ctx = _drive_set_time_step(loop, [])
    assert loop.keep_running is True
    assert loop.failed is False
    assert SolutionLoopPredicate()(ctx) is True


def test_predicate_stops_when_a_condition_is_satisfied() -> None:
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1))
    ctx = _drive_set_time_step(loop, [_stopper_runtime()])
    assert loop.keep_running is False
    assert loop.running() is True  # the loop stops even while running() is True
    assert SolutionLoopPredicate()(ctx) is False


def test_clean_stop_does_not_flag_failure() -> None:
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1))
    _drive_set_time_step(loop, [_stopper_runtime()])
    assert loop.stop_action == "end"
    assert loop.failed is False


def test_abort_stop_flags_failure() -> None:
    loop = SolutionLoop(state=_state(end=1.0, dt=0.1))
    _drive_set_time_step(loop, [_aborter_runtime()])
    assert loop.keep_running is False
    assert loop.stop_action == "abort"
    assert loop.failed is True


# --- the growth cap is the deciding term on the fold path -----------------


def test_constrain_delta_t_clamps_growth_to_the_cap() -> None:
    # A huge folded limit must be clamped to growth_cap * current, not applied raw.
    loop = SolutionLoop(state=_state(dt=0.1), growth_cap=1.2)
    loop.constrain_delta_t(10.0)
    assert loop.state.delta_t == pytest.approx(0.12)  # 1.2 * 0.1, not 10.0


# --- the published current step flows into a deltaT-taking contribution ----

_half_stepper = Model("halfStepper")


@_half_stepper.contributes(timeStepConstraint)
def _half_of_step(deltaT: float) -> float:
    return deltaT * 0.5


def _half_runtime() -> ModelRuntime:
    return ModelRuntime(spec=_half_stepper, name="halfStepper", config=None)


def test_published_step_flows_into_a_delta_t_contribution() -> None:
    # The driver publishes the loop's current step; a deltaT-taking contribution must
    # receive THAT value (0.2), so its 0.5x limit halves the step to 0.1.
    loop = SolutionLoop(state=_state(dt=0.2))
    _drive_set_time_step(loop, [_half_runtime()])
    assert loop.state.delta_t == pytest.approx(
        0.1
    )  # 0.5 * published 0.2, growth-cap ok


# --- legacy stepping symbols stay gone ------------------------------------


def test_install_constraints_step_is_absent() -> None:
    assert "install_constraints_step" not in dir(engine_mod)


def test_no_measurement_provider_nodes_in_the_loop_package() -> None:
    pkg_dir = Path(loop_pkg.__file__).parent
    sources = "\n".join(p.read_text() for p in pkg_dir.glob("*.py"))
    assert "measurement_provider" not in sources
