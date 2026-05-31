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

from typing import Any, cast

from neofoam.algorithms.constraints.time_step import (
    CourantConstraint,
    MaxDeltaTConstraint,
)
from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.control import SolutionControl
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
    # ending the run on convergence must not flip the state into a write step
    control = SolutionControl(residualControl={"p": 1e-2})
    loop = SolutionLoop(state=_state(end=100.0, dt=1.0), control=control)
    control.store_residual("p", 1e-3)  # converged
    assert loop.running() is False
    assert loop.state.write_time is False


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
    assert [s.name for s in steps] == ["time", "models.solution_loop"]

    state = steps[0].initializer({})
    assert isinstance(state, LoopState)

    assert "time" in steps[1].depends_on
    loop = steps[1].initializer({"time": state})
    assert isinstance(loop, SolutionLoop)
    assert loop.state is state
    assert {type(c).__name__ for c in loop.constraints} == {
        "CourantConstraint",
        "MaxDeltaTConstraint",
    }


# --- constraint seeding from config (make_solution_loop) ------------------


def test_adjustable_config_injects_courant_and_maxdeltat() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=0.5)
    loop = make_solution_loop(config, make_loop_state(config))
    assert {type(c).__name__ for c in loop.constraints} == {
        "CourantConstraint",
        "MaxDeltaTConstraint",
    }


def test_fixed_step_config_injects_no_constraints() -> None:
    loop = make_solution_loop(_config(adjustTimeStep=False), make_loop_state(_config()))
    assert loop.constraints == []


# --- injected Courant provider seam ---------------------------------------


def test_set_time_step_pushes_courant_from_injected_provider() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=10.0)
    loop = make_solution_loop(config, make_loop_state(config))
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
    assert loop.state.delta_t == 0.05  # 0.1 * 1.0 / 2.0 (CFL-limited)


def test_set_time_step_without_provider_just_adjusts() -> None:
    # no courant_provider injected: fixed step is left unchanged, no crash
    loop = make_solution_loop(_config(adjustTimeStep=False), make_loop_state(_config()))
    ctx = cast(Context, FakeContext({"solution_loop": loop}))
    set_time_step(None, ctx)
    assert loop.state.delta_t == 0.1


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
