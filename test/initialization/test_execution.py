# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for initialization execution functions."""

from typing import Any, Callable

import pytest

from neofoam.framework.initialization.execution import (
    InitializationGraphError,
    InitResult,
    build_context_from_objects,
    execute_initialization,
    execute_lazy_inits,
    topological_sort,
    validate,
)
from neofoam.framework.initialization.init_step import InitStep, InitStepExecutionError
from neofoam.framework.context import Context


# --- topological_sort ---


def test_sort_linear_chain(linear_chain: list[InitStep]) -> None:
    """A -> B -> C sorted correctly."""
    names = [li.name for li in topological_sort(linear_chain)]
    assert names == ["A", "B", "C"]


def test_sort_independent() -> None:
    """Independent nodes can appear in any valid order."""
    inits = [
        InitStep("X", depends_on=[], initializer=lambda _ctx: "x"),
        InitStep("Y", depends_on=[], initializer=lambda _ctx: "y"),
        InitStep("Z", depends_on=[], initializer=lambda _ctx: "z"),
    ]
    names = {li.name for li in topological_sort(inits)}
    assert names == {"X", "Y", "Z"}


def test_sort_diamond(diamond_graph: list[InitStep]) -> None:
    """A -> (B, C) -> D: A first, D last."""
    names = [li.name for li in topological_sort(diamond_graph)]
    assert names[0] == "A"
    assert names[-1] == "D"
    assert set(names[1:3]) == {"B", "C"}


def test_cycle_raises() -> None:
    """Circular A -> B -> C -> A raises ValueError."""
    inits = [
        InitStep("A", depends_on=["C"], initializer=lambda _ctx: "a"),
        InitStep("B", depends_on=["A"], initializer=lambda _ctx: "b"),
        InitStep("C", depends_on=["B"], initializer=lambda _ctx: "c"),
    ]
    with pytest.raises(ValueError, match="Circular dependency"):
        topological_sort(inits)


def test_missing_dep_raises() -> None:
    """Missing dependency raises ValueError."""
    inits = [InitStep("A", depends_on=["missing"], initializer=lambda _ctx: "a")]
    with pytest.raises(ValueError, match="depends on 'missing'"):
        topological_sort(inits)


def test_self_dependency_raises() -> None:
    """Self-dependency is detected as cycle."""
    inits = [InitStep("A", depends_on=["A"], initializer=lambda _ctx: "a")]
    with pytest.raises(ValueError, match="Circular dependency"):
        topological_sort(inits)


def test_duplicate_name_raises() -> None:
    """Duplicate InitStep names are rejected early."""
    inits = [
        InitStep("A", depends_on=[], initializer=lambda _ctx: "a1"),
        InitStep("A", depends_on=[], initializer=lambda _ctx: "a2"),
    ]
    with pytest.raises(ValueError, match="Duplicate InitStep name"):
        topological_sort(inits)


# --- execute_lazy_inits ---


def test_execute_lazy_inits_order() -> None:
    """Execution respects dependency order and passes context."""
    call_order: list[str] = []

    def _init_b(_ctx: object) -> str:
        call_order.append("B")
        return "b"

    def _init_a(_ctx: object) -> str:
        call_order.append("A")
        return "a"

    inits = [
        InitStep("B", depends_on=["A"], initializer=_init_b),
        InitStep("A", depends_on=[], initializer=_init_a),
    ]
    results = execute_lazy_inits(inits)
    assert call_order == ["A", "B"]
    assert {r.name: r.value for r in results} == {"A": "a", "B": "b"}


def test_execute_lazy_inits_context_passing() -> None:
    """Context dict is passed and grows as execution progresses."""
    inits = [
        InitStep("a", depends_on=[], initializer=lambda _ctx: 1),
        InitStep("b", depends_on=["a"], initializer=lambda ctx: ctx["a"] + 1),
        InitStep(
            "c", depends_on=["a", "b"], initializer=lambda ctx: ctx["a"] + ctx["b"]
        ),
    ]
    results = execute_lazy_inits(inits)
    values = {r.name: r.value for r in results}
    assert values == {"a": 1, "b": 2, "c": 3}


def test_execute_lazy_inits_returns_init_results() -> None:
    """execute_lazy_inits returns InitResult objects with category info."""
    inits = [
        InitStep(
            "fields.U", depends_on=[], initializer=lambda _ctx: "vel", category="fields"
        ),
        InitStep(
            "models.algo",
            depends_on=[],
            initializer=lambda _ctx: "alg",
            category="models",
        ),
    ]
    results = execute_lazy_inits(inits)
    assert all(isinstance(r, InitResult) for r in results)
    by_name = {r.name: r for r in results}
    assert by_name["fields.U"].category == "fields"
    assert by_name["fields.U"].value == "vel"
    assert by_name["models.algo"].category == "models"
    assert by_name["models.algo"].value == "alg"


# --- build_context_from_objects (parametrized) ---


@pytest.mark.parametrize(
    "init_results,check",
    [
        (
            [
                InitResult("fields.U", "fields", "v"),
                InitResult("fields.p", "fields", "p"),
            ],
            lambda c: c.fields == {"U": "v", "p": "p"},
        ),
        (
            [InitResult("models.algo", "models", "a")],
            lambda c: c.models == {"algo": "a"},
        ),
        (
            [InitResult("operators.mom", "operators", "m")],
            lambda c: c.models == {"mom": "m"},
        ),
        (
            [InitResult("mesh", "resource", "m")],
            lambda c: c.mesh == "m",
        ),
        (
            [InitResult("runtime", "resource", "r")],
            lambda c: c.runTime == "r",
        ),
        (
            [InitResult("custom", "resource", "c")],
            lambda c: c.models == {"custom": "c"},
        ),
    ],
    ids=["fields", "models", "operators", "mesh", "runtime", "unknown"],
)
def test_build_context_routing(
    init_results: list[InitResult], check: Callable[[Any], bool]
) -> None:
    """build_context_from_objects routes objects to correct Context slots."""
    ctx = build_context_from_objects(init_results)
    assert check(ctx)


# --- execute_initialization ---


def test_execute_initialization() -> None:
    """End-to-end: list[InitStep] -> Context."""
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "mesh_obj"),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda ctx: f"U_on_{ctx['mesh']}",
            category="fields",
        ),
        InitStep(
            "models.algo",
            depends_on=[],
            initializer=lambda _ctx: "algorithm",
            category="models",
        ),
    ]
    ctx = execute_initialization(inits)
    assert isinstance(ctx, Context)
    assert ctx.mesh == "mesh_obj"
    assert ctx.fields == {"U": "U_on_mesh_obj"}
    assert ctx.models == {"algo": "algorithm"}


def test_execute_initialization_complex() -> None:
    """Complex initialization with multi-level dependencies."""
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "mesh"),
        InitStep("runtime", depends_on=[], initializer=lambda _ctx: "runtime"),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda _ctx: "velocity",
            category="fields",
        ),
        InitStep(
            "fields.p",
            depends_on=["mesh"],
            initializer=lambda _ctx: "pressure",
            category="fields",
        ),
        InitStep(
            "operators.mom",
            depends_on=["fields.U", "fields.p"],
            initializer=lambda _ctx: "momentum_op",
            category="operators",
        ),
        InitStep(
            "models.algo",
            depends_on=["operators.mom"],
            initializer=lambda _ctx: "algorithm",
            category="models",
        ),
    ]
    ctx = execute_initialization(inits)
    assert ctx.mesh == "mesh"
    assert ctx.runTime == "runtime"
    assert len(ctx.fields) == 2
    assert len(ctx.models) == 2


def test_execute_initialization_raises_structured_graph_error() -> None:
    """execute_initialization raises InitializationGraphError with report."""
    inits = [InitStep("A", depends_on=["missing"], initializer=lambda _ctx: "a")]

    with pytest.raises(InitializationGraphError) as exc_info:
        execute_initialization(inits)

    report = exc_info.value.report
    assert not report.is_valid
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.code == "missing_dependency"
    assert diag.node_name == "A"


def test_execute_initialization_validates_only_once(monkeypatch: Any) -> None:
    """N1: execute_initialization should not re-validate during sorting."""
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "mesh_obj"),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda _ctx: "U",
            category="fields",
        ),
    ]

    validate_calls = {"count": 0}
    original_validate = validate

    def counting_validate(lazy_inits: list[InitStep]) -> object:
        validate_calls["count"] += 1
        return original_validate(lazy_inits)

    monkeypatch.setattr(
        "neofoam.framework.initialization.execution.validate", counting_validate
    )

    ctx = execute_initialization(inits)

    assert ctx.mesh == "mesh_obj"
    assert ctx.fields == {"U": "U"}
    assert validate_calls["count"] == 1


# --- validate ---


@pytest.mark.parametrize(
    "inits,expected_valid,expected_code,expected_msg_part",
    [
        (
            [
                InitStep("A", depends_on=["missing"], initializer=lambda _ctx: "a"),
                InitStep(
                    "B", depends_on=["also_missing"], initializer=lambda _ctx: "b"
                ),
            ],
            False,
            "missing_dependency",
            "depends on",
        ),
        (
            [
                InitStep("A", depends_on=["B"], initializer=lambda _ctx: "a"),
                InitStep("B", depends_on=["A"], initializer=lambda _ctx: "b"),
            ],
            False,
            "cycle",
            "Circular dependency",
        ),
        (
            [
                InitStep("dup", depends_on=[], initializer=lambda _ctx: 1),
                InitStep("dup", depends_on=[], initializer=lambda _ctx: 2),
            ],
            False,
            "duplicate_name",
            "Duplicate InitStep name",
        ),
        (
            [
                InitStep("A", depends_on=[], initializer=lambda _ctx: "a"),
                InitStep("B", depends_on=["A"], initializer=lambda _ctx: "b"),
            ],
            True,
            None,
            None,
        ),
    ],
    ids=["missing", "cycle", "duplicate", "clean"],
)
def test_validate_graph(
    inits: list[InitStep],
    expected_valid: bool,
    expected_code: str | None,
    expected_msg_part: str | None,
) -> None:
    """validate() returns structured graph report diagnostics."""
    report = validate(inits)
    assert report.is_valid is expected_valid
    if expected_valid:
        assert report.diagnostics == ()
        return

    assert len(report.diagnostics) >= 1
    first = report.diagnostics[0]
    assert first.code == expected_code
    assert expected_msg_part is not None
    assert expected_msg_part in first.message


def test_build_context_routes_by_category() -> None:
    """build_context_from_objects uses category when available."""
    init_results = [
        InitResult("fields.U", "fields", "velocity"),
        InitResult("models.transport", "models", "transport_model"),
        InitResult("operators.laplacian", "operators", "laplacian_op"),
    ]
    ctx = build_context_from_objects(init_results)
    assert ctx.fields == {"U": "velocity"}
    assert ctx.models == {
        "transport": "transport_model",
        "laplacian": "laplacian_op",
    }


def test_build_context_category_without_prefix() -> None:
    """Category routing works even without a matching name prefix."""
    init_results = [
        InitResult("velocity", "fields", "vel_value"),
        InitResult("algo", "models", "algo_value"),
    ]
    ctx = build_context_from_objects(init_results)
    assert ctx.fields == {"velocity": "vel_value"}
    assert ctx.models == {"algo": "algo_value"}


def test_build_context_resource_routing() -> None:
    """Resource category routes mesh/runtime specially and others to models."""
    init_results = [
        InitResult("mesh", "resource", "mesh_obj"),
        InitResult("runtime", "resource", "runtime_obj"),
        InitResult("solver_state", "resource", "state_obj"),
    ]
    ctx = build_context_from_objects(init_results)
    assert ctx.mesh == "mesh_obj"
    assert ctx.runTime == "runtime_obj"
    assert ctx.models["solver_state"] == "state_obj"


def test_init_step_execution_error_wraps_cause() -> None:
    """InitStep.execute() wraps initializer exceptions with step context."""

    def failing_init(_ctx: object) -> None:
        raise RuntimeError("disk read failed")

    step = InitStep("fields.U", depends_on=["mesh"], initializer=failing_init)

    with pytest.raises(InitStepExecutionError) as exc_info:
        step.execute({})

    err = exc_info.value
    assert err.step_name == "fields.U"
    assert err.depends_on == ["mesh"]
    assert "disk read failed" in str(err)
    assert err.__cause__ is not None


def test_init_step_execution_error_preserves_value_error() -> None:
    """ValueError from initializer is not wrapped."""
    step = InitStep(
        "value_err",
        initializer=lambda _ctx: (_ for _ in ()).throw(ValueError("bad value")),
    )
    with pytest.raises(ValueError, match="bad value"):
        step.execute(context={})


def test_init_step_execution_error_in_pipeline() -> None:
    """InitStepExecutionError propagates through execute_lazy_inits."""

    def boom(ctx: object) -> None:
        raise OSError("cannot open file")

    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "m"),
        InitStep("fields.U", depends_on=["mesh"], initializer=boom),
    ]

    with pytest.raises(InitStepExecutionError) as exc_info:
        execute_lazy_inits(inits)

    assert exc_info.value.step_name == "fields.U"
