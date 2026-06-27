# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the runTimeControl loopCondition contribution (solver-side, pybFoam)."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import importlib
import inspect
import shutil
from pathlib import Path
from typing import Any

import pydantic
import pytest

pytest.importorskip("pybFoam")

from neofoam.algorithms.solution_loop.conditions import fold_conditions
from neofoam.algorithms.solution_loop.interfaces import loopCondition
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.solution_loop import (
    SolutionLoop,
    SolutionLoopPredicate,
)
from neofoam.algorithms.solution_loop.solution_loop import (
    set_time_step,
    solutionLoop,
)
from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.model import (
    BoundModelInterface,
    ModelRuntime,
    bind_owned_interfaces,
)
from neofoam.io.strategies.json_strategy import JSONStrategy
from neofoam.io.strategies.openfoam_strategy import OpenFOAMStrategy
from neofoam.io.strategies.yaml_strategy import YAMLStrategy
from neofoam.solver.incompressibleFluid.models.incompressibleFluidModel import (
    incompressibleFluidModel,
)
from neofoam.solver.incompressibleFluid.models.run_time_control import (
    ConditionContext,
    EquationInitialResidualCondition,
    MinMaxCondition,
    RunTimeCondition,
    RunTimeControlConfig,
    detect_residual_control,
    evaluate_conditions,
    residual_conditions,
    residualControl,
    runTimeControl,
)

rtc = importlib.import_module(
    "neofoam.solver.incompressibleFluid.models.run_time_control"
)

_CASES = Path(__file__).parent / "cases"


def _control_runtime(config: RunTimeControlConfig) -> ModelRuntime:
    return ModelRuntime(spec=runTimeControl, name="runTimeControl", config=config)


# --- MinMaxCondition --------------------------------------------------------


def test_min_max_maximum_triggers_above_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda f: [10.0, 600.0])
    condition = MinMaxCondition(mode="maximum", fields=["U"], value=500.0)
    vote = condition.vote(ConditionContext(fields={"U": object()}))
    assert vote.satisfied is True
    assert vote.group_id == -1


def test_min_max_maximum_does_not_trigger_below_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda f: [10.0, 90.0])
    condition = MinMaxCondition(mode="maximum", fields=["U"], value=500.0)
    assert condition.vote(ConditionContext(fields={"U": object()})).satisfied is False


def test_min_max_minimum_mode_uses_min_and_less_than(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda f: [2.0, 0.5])
    cctx = ConditionContext(fields={"U": object()})
    assert (
        MinMaxCondition(mode="minimum", fields=["U"], value=1.0).vote(cctx).satisfied
        is True
    )
    assert (
        MinMaxCondition(mode="minimum", fields=["U"], value=0.1).vote(cctx).satisfied
        is False
    )


def test_min_max_carries_group_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda f: [600.0])
    condition = MinMaxCondition(mode="maximum", fields=["U"], value=500.0, groupID=2)
    assert condition.vote(ConditionContext(fields={"U": object()})).group_id == 2


def test_min_max_aggregates_over_multiple_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mags = {"u": [10.0], "p": [600.0]}
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda obj: mags[obj])
    condition = MinMaxCondition(mode="maximum", fields=["U", "p"], value=500.0)
    cctx = ConditionContext(fields={"U": "u", "p": "p"})
    assert condition.vote(cctx).satisfied is True  # max(10, 600) > 500


def test_min_max_strict_boundary_excludes_equal_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda obj: [500.0])
    condition = MinMaxCondition(mode="maximum", fields=["U"], value=500.0)
    assert condition.vote(ConditionContext(fields={"U": object()})).satisfied is False


# --- EquationInitialResidualCondition ---------------------------------------


@pytest.mark.parametrize(
    "residual, expected",
    [(1e-5, True), (1e-4, True), (1e-3, False)],
)
def test_residual_condition_compares_against_tolerance(
    residual: float, expected: bool
) -> None:
    condition = EquationInitialResidualCondition(fields=["U"], value=1e-4)
    vote = condition.vote(ConditionContext(residuals={"U": residual}))
    assert vote.satisfied is expected


def test_residual_condition_missing_field_is_not_satisfied() -> None:
    condition = EquationInitialResidualCondition(fields=["U"], value=1e-4)
    assert condition.vote(ConditionContext(residuals={})).satisfied is False


@pytest.mark.parametrize(
    "residuals, expected",
    [
        ({"U": 1e-5, "p": 1e-2}, False),
        ({"U": 1e-5, "p": 1e-5}, True),
    ],
)
def test_residual_condition_and_over_all_fields(
    residuals: dict[str, float], expected: bool
) -> None:
    condition = EquationInitialResidualCondition(fields=["U", "p"], value=1e-4)
    assert condition.vote(ConditionContext(residuals=residuals)).satisfied is expected


# --- residual_conditions normalization --------------------------------------


def test_residual_conditions_makes_one_condition_per_field() -> None:
    conds = residual_conditions({"p": 1e-2, "U": 1e-3}, group_id=0)
    assert len(conds) == 2
    by_field = {c.fields[0]: c for c in conds}
    assert isinstance(by_field["p"], EquationInitialResidualCondition)
    assert by_field["p"].value == pytest.approx(1e-2)
    assert by_field["U"].value == pytest.approx(1e-3)


def test_residual_conditions_share_one_and_group() -> None:
    conds = residual_conditions({"p": 1e-2, "U": 1e-3}, group_id=0)
    assert {c.groupID for c in conds} == {0}


def test_residual_group_satisfied_only_when_every_field_converges() -> None:
    conds = residual_conditions({"p": 1e-2, "U": 1e-3}, group_id=0)
    mixed = ConditionContext(residuals={"p": 1e-3, "U": 1e-1})  # U above tol
    assert fold_conditions(c.vote(mixed) for c in conds).satisfied is False
    converged = ConditionContext(residuals={"p": 1e-3, "U": 1e-4})  # all below
    vote = fold_conditions(c.vote(converged) for c in conds)
    assert vote.satisfied is True
    assert vote.action == "end"  # never abort


# --- residual convergence ends the run via the fold -------------------------


def _loop_state(end: float = 100.0, dt: float = 1.0) -> LoopState:
    return LoopState(
        value=0.0,
        delta_t=dt,
        end_time=end,
        write_control="timeStep",
        write_interval=1.0,
    )


def test_residual_convergence_ends_the_run_through_the_fold() -> None:
    conds = residual_conditions({"p": 1e-2, "U": 1e-3}, group_id=0)
    cctx = ConditionContext(residuals={"p": 1e-3, "U": 1e-4})  # all converged
    vote = fold_conditions(c.vote(cctx) for c in conds)
    loop = SolutionLoop(state=_loop_state())
    loop.keep_running = not vote.satisfied
    ctx = Context(fields={}, models={"solution_loop": loop})
    assert loop.running() is True  # the run would otherwise continue
    assert SolutionLoopPredicate()(ctx) is False  # ...but the fold ends it
    assert loop.failed is False  # clean end, not a failure


def test_partial_residual_convergence_keeps_running_through_the_fold() -> None:
    conds = residual_conditions({"p": 1e-2, "U": 1e-3}, group_id=0)
    cctx = ConditionContext(residuals={"p": 1e-3, "U": 1e-1})  # U not converged
    vote = fold_conditions(c.vote(cctx) for c in conds)
    loop = SolutionLoop(state=_loop_state())
    loop.keep_running = not vote.satisfied
    ctx = Context(fields={}, models={"solution_loop": loop})
    assert SolutionLoopPredicate()(ctx) is True


# --- RunTimeControlConfig validation ----------------------------------------


def test_unknown_condition_type_is_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        RunTimeControlConfig(conditions={"x": {"type": "bogus", "value": 1.0}})
    with pytest.raises(pydantic.ValidationError):
        RunTimeCondition.create(condition={"type": "bogus"})


def test_known_condition_type_round_trips() -> None:
    cfg = RunTimeControlConfig(
        conditions={
            "d": {"type": "minMax", "mode": "maximum", "fields": ["U"], "value": 500.0}
        }
    )
    assert isinstance(cfg.conditions["d"], MinMaxCondition)
    assert cfg.satisfiedAction == "end"


# --- placement + family registration ----------------------------------------


def test_contribution_lives_under_the_solver_not_the_framework() -> None:
    assert evaluate_conditions.__module__.startswith(
        "neofoam.solver.incompressibleFluid"
    )


def test_model_is_registered_in_the_family_catalog() -> None:
    names = {spec.name for spec in incompressibleFluidModel.all_specs()}
    assert "runTimeControl" in names


# --- gating through the bound interface --------------------------------------


def test_active_model_folds_its_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda f: [600.0])
    config = RunTimeControlConfig(
        conditions={"div": MinMaxCondition(mode="maximum", fields=["U"], value=500.0)},
        satisfiedAction="abort",
    )
    ctx = Context(fields={"U": object()}, models={})
    bound = BoundModelInterface(loopCondition, [_control_runtime(config)], ctx)
    result = bound()
    assert result.satisfied is True
    assert result.action == "abort"


def test_inactive_model_is_excluded_from_the_fold() -> None:
    ctx = Context(fields={"U": object()}, models={})
    bound = BoundModelInterface(loopCondition, [], ctx)  # runTimeControl not active
    assert bound().satisfied is False


def test_inactive_condition_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda f: [600.0])
    config = RunTimeControlConfig(
        conditions={
            "div": MinMaxCondition(
                mode="maximum", fields=["U"], value=500.0, active=False
            )
        }
    )
    ctx = Context(fields={"U": object()}, models={})
    bound = BoundModelInterface(loopCondition, [_control_runtime(config)], ctx)
    assert bound().satisfied is False


def test_unresolvable_param_raises_at_fold_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rtc, "_field_magnitudes", lambda f: [600.0])
    config = RunTimeControlConfig(
        conditions={"div": MinMaxCondition(mode="maximum", fields=["U"], value=500.0)}
    )
    bound = BoundModelInterface(
        loopCondition, [_control_runtime(config)], Context(fields={}, models={})
    )
    with pytest.raises(
        ValueError,
        match=r"interface 'loopCondition'.*'evaluate_conditions'.*parameter 'U'.*no provider supplies it",
    ):
        bound()


def test_contribution_never_declares_a_context_param() -> None:
    params = set(inspect.signature(evaluate_conditions).parameters)
    assert "ctx" not in params
    assert "Context" not in params


# --- dict loading: the nested conditions block (JSON/YAML proven path) -------


def test_run_time_control_block_loads_one_condition_per_subdict_key() -> None:
    path = _CASES / "runTimeControl" / "conditions.json"
    data = JSONStrategy().read(RunTimeControlConfig, path)
    cfg = RunTimeControlConfig.model_validate(data)
    assert set(cfg.conditions) == {"divergence", "uResidual"}
    assert isinstance(cfg.conditions["divergence"], MinMaxCondition)
    assert isinstance(cfg.conditions["uResidual"], EquationInitialResidualCondition)


def test_omitted_group_id_and_active_take_their_defaults() -> None:
    path = _CASES / "runTimeControl" / "conditions.json"
    cfg = RunTimeControlConfig.model_validate(
        JSONStrategy().read(RunTimeControlConfig, path)
    )
    divergence = cfg.conditions["divergence"]
    assert divergence.groupID == -1
    assert divergence.active is True


def test_omitted_satisfied_action_defaults_to_end() -> None:
    path = _CASES / "runTimeControl" / "conditions.json"
    cfg = RunTimeControlConfig.model_validate(
        JSONStrategy().read(RunTimeControlConfig, path)
    )
    assert cfg.satisfiedAction == "end"


def test_yaml_and_json_yield_an_equal_config() -> None:
    base = _CASES / "runTimeControl"
    json_cfg = RunTimeControlConfig.model_validate(
        JSONStrategy().read(RunTimeControlConfig, base / "conditions.json")
    )
    yaml_cfg = RunTimeControlConfig.model_validate(
        YAMLStrategy().read(RunTimeControlConfig, base / "conditions.yaml")
    )
    assert yaml_cfg == json_cfg
    # Independent of the equality: the YAML side discriminates the subtype + default.
    assert isinstance(yaml_cfg.conditions["divergence"], MinMaxCondition)
    assert yaml_cfg.conditions["divergence"].groupID == -1


def test_openfoam_reader_cannot_read_the_nested_conditions_dict() -> None:
    # Documented limitation: the OpenFOAM dictionary reader is scalar/single-BaseModel
    # only, so the arbitrary-keyed `conditions` sub-dict cannot be read from a real
    # controlDict yet. JSON/YAML is the loading path until the reader is extended.
    control_dict = _CASES / "runTimeControl" / "system" / "controlDict"
    strategy = OpenFOAMStrategy("functions.runTimeControl1")
    with pytest.raises(TypeError, match="conditions"):
        strategy.read(RunTimeControlConfig, control_dict)


# --- detect_model activation gate -------------------------------------------


def test_detect_inactive_when_no_control_dict_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert runTimeControl.run_detect() is False


def test_detect_inactive_when_control_dict_has_no_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "maxCo_present")  # fixed-step case, no functions block
    assert runTimeControl.run_detect() is False


def test_detect_active_for_a_run_time_control_function_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "runTimeControl")
    assert runTimeControl.run_detect() is True


def test_detect_inactive_for_a_different_function_object_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "runTimeControl_absent")
    assert runTimeControl.run_detect() is False


# --- live pybFoam reduction --------------------------------------------------


def test_field_magnitudes_reduces_a_live_vol_vector_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pybFoam as pyf
    from pybFoam import volVectorField

    # Stage the checked-in mesh + field in a scratch case (read is destructive of the
    # latest-time discovery, so copy rather than read in place) and build a real field.
    for sub in ("system", "constant", "0"):
        shutil.copytree(_CASES / "live_field" / sub, tmp_path / sub)
    monkeypatch.chdir(tmp_path)
    # Keep argList/Time/mesh as separate live references — a single nested
    # expression frees the temporaries and segfaults on field read.
    args = pyf.argList(["test"])
    runtime = pyf.Time(args)
    mesh = pyf.fvMesh(runtime)
    U = volVectorField.read_field(mesh, "U")

    magnitudes = rtc._field_magnitudes(U)
    assert len(magnitudes) == 25  # the 25-cell fixture mesh
    assert all(m == pytest.approx(3.0**0.5) for m in magnitudes)  # |(1,1,1)| = sqrt 3

    # The same live field drives the MinMaxCondition end-to-end (not a stub).
    vote = MinMaxCondition(mode="maximum", fields=["U"], value=1.0).vote(
        ConditionContext(fields={"U": U})
    )
    assert vote.satisfied is True


# --- MinMaxCondition named lookup error --------------------------------------


def test_min_max_raises_named_error_for_a_missing_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = MinMaxCondition(mode="maximum", fields=["U"], value=500.0)
    with pytest.raises(ValueError, match=r"minMax.*field 'U'.*available"):
        condition.vote(ConditionContext(fields={}))


# --- abstract base condition guard -------------------------------------------


def test_base_condition_vote_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        RunTimeCondition().vote(ConditionContext())


# --- residualControl model: discovery / detect / load / build ----------------


def test_residual_control_is_registered_in_the_family_catalog() -> None:
    names = {spec.name for spec in incompressibleFluidModel.all_specs()}
    assert "residualControl" in names


def test_residual_control_detects_a_fvsolution_residual_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(_CASES / "residualControl_simple")
    assert detect_residual_control() is True


def test_residual_control_load_reads_the_residualcontrol_into_conditions() -> None:
    rt = residualControl.instantiate(_CASES / "residualControl_simple")
    cfg = rt.config
    assert all(isinstance(c, EquationInitialResidualCondition) for c in cfg.conditions)
    by_field = {c.fields[0]: c.value for c in cfg.conditions}
    assert by_field["p"] == pytest.approx(1e-2)
    assert by_field["U"] == pytest.approx(1e-3)
    assert {c.groupID for c in cfg.conditions} == {0}  # one AND group


def test_residual_control_build_publishes_an_empty_residual_store() -> None:
    rt = residualControl.instantiate(_CASES / "residualControl_simple")
    steps = {s.name: s for s in rt.run_build()}
    assert "models.residuals" in steps
    assert steps["models.residuals"].initializer({}) == {}


# --- residualControl ends the run through the loopCondition fold --------------


def _drive_residual_set_time_step(
    loop: SolutionLoop, res_rt: ModelRuntime, residuals: dict[str, float]
) -> Context:
    loop_rt = ModelRuntime(spec=solutionLoop, name="solutionLoop", config=None)
    models: dict[str, Any] = {
        "solution_loop": loop,
        "solutionLoop": loop_rt,
        "residualControl": res_rt,
        "residuals": residuals,
    }
    ctx = Context(fields={}, models=models)
    bind_owned_interfaces(loop_rt, [res_rt], ctx)
    wrap_with_dependency_resolution(
        set_time_step, instance=None, dependency_resolver=DependencyResolver()
    )(ctx)
    return ctx


def test_residual_control_ends_the_run_through_the_loop_condition() -> None:
    res_rt = residualControl.instantiate(_CASES / "residualControl_simple")
    loop = SolutionLoop(state=_loop_state())
    # injected stand-in for live-published residuals (solve capture deferred):
    converged = {"p": 1e-5, "U": 1e-6, "(k|epsilon)": 1e-6}
    ctx = _drive_residual_set_time_step(loop, res_rt, converged)
    assert loop.keep_running is False
    assert loop.running() is True  # would otherwise still be running
    assert loop.failed is False  # clean "end", never abort
    assert SolutionLoopPredicate()(ctx) is False


def test_residual_control_keeps_running_until_every_field_converges() -> None:
    res_rt = residualControl.instantiate(_CASES / "residualControl_simple")
    loop = SolutionLoop(state=_loop_state())
    partial = {"p": 1e-5, "U": 1e-1, "(k|epsilon)": 1e-6}  # U above tolerance
    ctx = _drive_residual_set_time_step(loop, res_rt, partial)
    assert loop.keep_running is True
    assert SolutionLoopPredicate()(ctx) is True


def test_residual_control_is_a_noop_stop_source_with_an_empty_store() -> None:
    # parity-safety: until live publishing lands, an empty store never stops the run.
    res_rt = residualControl.instantiate(_CASES / "residualControl_simple")
    loop = SolutionLoop(state=_loop_state())
    ctx = _drive_residual_set_time_step(loop, res_rt, {})
    assert loop.keep_running is True
    assert SolutionLoopPredicate()(ctx) is True
