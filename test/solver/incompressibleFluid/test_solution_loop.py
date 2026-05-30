# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the solutionLoop Model wiring.

Pure-Python: the stepper (FoamTime) does advancement; a fake backend stands in
for pybFoam.Time at the StepSink seam. No OpenFOAM case needed.
"""

from __future__ import annotations

from typing import cast

from neofoam.algorithms.time_step import CourantConstraint
from neofoam.framework.context import Context
from neofoam.algorithms.foam_time import FoamTime
from neofoam.algorithms.solution_loop import SolutionLoop
from neofoam.solver.incompressibleFluid.configs import ControlDictConfig
from neofoam.solver.incompressibleFluid.models.solution_loop import (
    PybFoamStepSink,
    SolutionLoopPredicate,
    build,
    increment_time,
    loop_backend_steps,
    make_solution_loop,
    make_stepper,
    solutionLoop,
)


def _config(**kw: object) -> ControlDictConfig:
    base: dict[str, object] = {"endTime": 0.3, "deltaT": 0.1}
    base.update(kw)
    return ControlDictConfig(**base)


# --- the Model ------------------------------------------------------------


def test_modelspec_is_a_full_model() -> None:
    assert solutionLoop.name == "solutionLoop"
    assert solutionLoop._load_func is not None
    assert solutionLoop._build_func is not None
    op_names = {meta["name"] for _, meta in solutionLoop._operations}
    # writing is the fieldWriter Model's concern, not the loop's
    assert op_names == {"set_time_step", "increment_time"}


def test_build_emits_stepper_then_engine() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=0.5)
    steps = build(config)
    assert [s.name for s in steps] == ["models.stepper", "models.solution_loop"]

    # stepper step builds the pure-Python FoamTime
    stepper = steps[0].initializer({"runtime": FakeRuntime()})
    assert isinstance(stepper, FoamTime)

    # engine step wraps that stepper and seeds constraints from config
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


def test_adjustable_without_maxdeltat_injects_only_courant() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0)
    loop = make_solution_loop(config, make_stepper(config))
    assert {type(c) for c in loop.constraints} == {CourantConstraint}


def test_fixed_step_config_injects_no_constraints() -> None:
    loop = make_solution_loop(_config(adjustTimeStep=False), make_stepper(_config()))
    assert loop.constraints == []


def test_courant_constraint_shrinks_step() -> None:
    config = _config(adjustTimeStep=True, maxCo=1.0, maxDeltaT=10.0)
    loop = make_solution_loop(config, make_stepper(config))
    loop.set_courant(2.0)
    loop.adjust_delta_t()
    assert loop.stepper.deltaTValue() == 0.05  # 0.1 * 1.0 / 2.0


# --- PybFoamStepSink: mirror the stepper onto the backend -----------------


class FakeRuntime:
    """Stands in for pybFoam.Time at the StepSink seam (setDeltaT + increment)."""

    def __init__(self) -> None:
        self.delta_t: float = 0.0
        self.steps = 0

    def setDeltaT(self, dt: float) -> None:
        self.delta_t = dt

    def increment(self) -> None:
        self.steps += 1


def test_step_sink_forwards_to_backend() -> None:
    rt = FakeRuntime()
    sink = PybFoamStepSink(rt)
    sink.set_delta_t(0.2)
    sink.advance_to(0.5, 5)
    assert rt.delta_t == 0.2
    assert rt.steps == 1  # advanced the backend by one step


def test_stepper_syncs_backend_through_injected_sink() -> None:
    rt = FakeRuntime()
    stepper = make_stepper(_config(deltaT=0.1))
    stepper.set_sink(PybFoamStepSink(rt))  # backend sink injected post-build
    stepper.setDeltaT(0.1)  # pushes deltaT to the backend
    stepper.increment()  # advances both the stepper and the backend
    assert rt.delta_t == 0.1
    assert rt.steps == 1


def test_loop_backend_steps_inject_sink_courant_and_logger() -> None:
    rt = FakeRuntime()
    stepper = make_stepper(_config(deltaT=0.1))
    steps = loop_backend_steps()
    by_name = {s.name: s for s in steps}
    assert set(by_name) == {
        "models.stepper_sink",
        "models.courant_provider",
        "models.loop_logger",
    }

    # the sink step mutates the framework-built stepper, then it mirrors steps
    ctx = {"models.stepper": stepper, "runtime": rt}
    by_name["models.stepper_sink"].initializer(ctx)
    stepper.setDeltaT(0.2)
    stepper.increment()
    assert rt.delta_t == 0.2
    assert rt.steps == 1

    # courant_provider + loop_logger are registered as plain callables
    provider = by_name["models.courant_provider"].initializer(ctx)
    assert callable(provider)
    logger = by_name["models.loop_logger"].initializer(ctx)
    assert callable(logger)


# --- predicate + operation delegation -------------------------------------


class FakeContext:
    def __init__(self, loop: SolutionLoop) -> None:
        self.models = {"solution_loop": loop}


def test_predicate_delegates_to_engine_running() -> None:
    on = make_solution_loop(_config(endTime=1.0), make_stepper(_config(endTime=1.0)))
    off = make_solution_loop(_config(endTime=1.0), make_stepper(_config(endTime=1.0)))
    off.stepper.stop()
    assert SolutionLoopPredicate()(cast(Context, FakeContext(on))) is True
    assert SolutionLoopPredicate()(cast(Context, FakeContext(off))) is False


def test_increment_time_advances_stepper() -> None:
    loop = make_solution_loop(_config(), make_stepper(_config()))
    ctx = cast(Context, FakeContext(loop))
    increment_time(None, ctx)
    assert loop.stepper.timeIndex() == 1
