# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the solutionLoop Model wiring.

Pure-Python: SolutionLoop advances a LoopState; a fake stands in for pybFoam.Time
at the FoamTime LoopBackend seam. No OpenFOAM case needed.
"""

from __future__ import annotations

from typing import cast

import pytest

from neofoam.algorithms.constraints.time_step import CourantConstraint
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.solution_loop import (
    SolutionLoop,
    update_loop_controls,
)
from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.model import ModelRuntime, bind_owned_interfaces
from neofoam.solver.incompressibleFluid.configs import ControlDictConfig
from neofoam.solver.incompressibleFluid.models.max_delta_t import (
    MaxDeltaTConfig,
    maxDeltaT,
)
from neofoam.solver.incompressibleFluid.models.solution_loop import (
    FoamTime,
    SolutionLoopPredicate,
    build,
    increment_time,
    loop_backend_steps,
    make_loop_state,
    make_solution_loop,
    solutionLoop,
)


def _config(**kw: object) -> ControlDictConfig:
    base: dict[str, object] = {"endTime": 0.3, "deltaT": 0.1}
    base.update(kw)
    return ControlDictConfig(**base)


class FakeRuntime:
    """Stands in for pybFoam.Time at the FoamTime backend seam (setDeltaT + increment)."""

    def __init__(self) -> None:
        self.delta_t: float = 0.0
        self.steps = 0

    def setDeltaT(self, dt: float) -> None:
        self.delta_t = dt

    def increment(self) -> None:
        self.steps += 1


# --- the Model ------------------------------------------------------------


def test_modelspec_is_a_full_model() -> None:
    assert solutionLoop.name == "solutionLoop"
    assert solutionLoop._load_func is not None
    assert solutionLoop._build_func is not None
    op_names = {meta["name"] for _, meta in solutionLoop._operations}
    # writing is the fieldWriter Model's concern, not the loop's
    assert op_names == {"set_time_step", "increment_time", "update_loop_controls"}


def test_build_emits_state_then_engine() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=0.5)
    steps = build(config)
    assert [s.name for s in steps] == [
        "time",
        "models.solution_loop",
    ]

    # the "time" step builds the pure-Python LoopState (ctx.time)
    state = steps[0].initializer({})
    assert isinstance(state, LoopState)

    # engine step wraps that state and seeds constraints from config
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


def test_adjustable_without_maxdeltat_injects_only_courant() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0)
    loop = make_solution_loop(config, make_loop_state(config))
    assert {type(c) for c in loop.constraints} == {CourantConstraint}


def test_fixed_step_config_injects_no_constraints() -> None:
    loop = make_solution_loop(_config(adjustTimeStep=False), make_loop_state(_config()))
    assert loop.constraints == []


def test_courant_constraint_shrinks_step() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=10.0)
    loop = make_solution_loop(config, make_loop_state(config))
    loop.set_courant(2.0)
    loop.adjust_delta_t()
    assert loop.state.delta_t == 0.05  # 0.1 * 1.0 / 2.0


# --- FoamTime LoopBackend: mirror the LoopState onto pybFoam.Time ---------


def test_foam_time_backend_reconciles_pyf_time() -> None:
    rt = FakeRuntime()
    backend = FoamTime(rt)
    backend.update(LoopState(value=0.0, delta_t=0.2, end_time=1.0, index=0))
    assert rt.delta_t == 0.2
    assert rt.steps == 0  # index 0 -> no increment yet
    backend.update(LoopState(value=0.2, delta_t=0.2, end_time=1.0, index=1))
    assert rt.steps == 1  # advanced the backend by one step to match index 1


def test_loop_backend_steps_inject_backend_courant_and_logger() -> None:
    rt = FakeRuntime()
    loop = make_solution_loop(_config(deltaT=0.1), make_loop_state(_config(deltaT=0.1)))
    steps = loop_backend_steps()
    by_name = {s.name: s for s in steps}
    assert set(by_name) == {
        "models.loop_backend",
        "models.courant_provider",
        "models.loop_logger",
    }

    # the backend step injects FoamTime into the engine; advancing now mirrors
    ctx = {"models.solution_loop": loop, "_foam_time": rt}
    by_name["models.loop_backend"].initializer(ctx)
    loop.advance()
    assert rt.steps == 1
    assert rt.delta_t == loop.state.delta_t

    provider = by_name["models.courant_provider"].initializer(ctx)
    assert callable(provider)
    logger = by_name["models.loop_logger"].initializer(ctx)
    assert callable(logger)


# --- predicate + operation delegation -------------------------------------


class FakeContext:
    def __init__(self, loop: SolutionLoop) -> None:
        self.models = {"solution_loop": loop}


def test_predicate_delegates_to_engine_running() -> None:
    on = make_solution_loop(_config(endTime=1.0), make_loop_state(_config(endTime=1.0)))
    off = make_solution_loop(
        _config(endTime=1.0), make_loop_state(_config(endTime=1.0))
    )
    off.stop()
    assert SolutionLoopPredicate()(cast(Context, FakeContext(on))) is True
    assert SolutionLoopPredicate()(cast(Context, FakeContext(off))) is False


def test_increment_time_advances_state() -> None:
    loop = make_solution_loop(_config(), make_loop_state(_config()))
    ctx = cast(Context, FakeContext(loop))
    increment_time(None, ctx)
    assert loop.state.index == 1


def test_active_optional_model_limit_flows_through_update_loop_controls() -> None:
    # Mirrors the live wiring create_fields installs: the solutionLoop owner runtime
    # has the case's active maxDeltaT contributor bound onto it, so that owned
    # contribution folds through update_loop_controls into the engine's next_dt.
    loop = make_solution_loop(_config(), make_loop_state(_config()))
    loop.next_dt = 999.0  # sentinel distinct from the fold result
    loop_rt = ModelRuntime(spec=solutionLoop, name="solutionLoop", config=None)
    max_rt = ModelRuntime(
        spec=maxDeltaT, name="maxDeltaT", config=MaxDeltaTConfig(maxDeltaT=0.5)
    )
    ctx = Context(
        fields={},
        models={"solution_loop": loop, "solutionLoop": loop_rt, "maxDeltaT": max_rt},
    )
    bind_owned_interfaces(loop_rt, [max_rt], ctx)
    wrapped = wrap_with_dependency_resolution(
        update_loop_controls, instance=None, dependency_resolver=DependencyResolver()
    )
    wrapped(ctx)
    assert loop.next_dt == pytest.approx(0.5)
