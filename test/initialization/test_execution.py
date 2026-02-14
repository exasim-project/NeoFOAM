# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for initialization execution functions."""

import pytest

from neofoam.framework.initialization.execution import (
    InitGraphValidator,
    InitializationGraphError,
    topological_sort,
    execute_lazy_inits,
    build_context_from_objects,
    execute_initialization,
    validate_lazy_init_graph,
)
from neofoam.framework.initialization.init_step import InitStep
from neofoam.framework.context import Context


# --- topological_sort ---


def test_sort_linear_chain(linear_chain):
    """A -> B -> C sorted correctly."""
    names = [li.name for li in topological_sort(linear_chain)]
    assert names == ["A", "B", "C"]


def test_sort_independent():
    """Independent nodes can appear in any valid order."""
    inits = [
        InitStep("X", depends_on=[], initializer=lambda: "x"),
        InitStep("Y", depends_on=[], initializer=lambda: "y"),
        InitStep("Z", depends_on=[], initializer=lambda: "z"),
    ]
    names = {li.name for li in topological_sort(inits)}
    assert names == {"X", "Y", "Z"}


def test_sort_diamond(diamond_graph):
    """A -> (B, C) -> D: A first, D last."""
    names = [li.name for li in topological_sort(diamond_graph)]
    assert names[0] == "A"
    assert names[-1] == "D"
    assert set(names[1:3]) == {"B", "C"}


def test_cycle_raises():
    """Circular A -> B -> C -> A raises ValueError."""
    inits = [
        InitStep("A", depends_on=["C"], initializer=lambda: "a"),
        InitStep("B", depends_on=["A"], initializer=lambda: "b"),
        InitStep("C", depends_on=["B"], initializer=lambda: "c"),
    ]
    with pytest.raises(ValueError, match="Circular dependency"):
        topological_sort(inits)


def test_missing_dep_raises():
    """Missing dependency raises ValueError."""
    inits = [InitStep("A", depends_on=["missing"], initializer=lambda: "a")]
    with pytest.raises(ValueError, match="depends on 'missing'"):
        topological_sort(inits)


def test_self_dependency_raises():
    """Self-dependency is detected as cycle."""
    inits = [InitStep("A", depends_on=["A"], initializer=lambda: "a")]
    with pytest.raises(ValueError, match="Circular dependency"):
        topological_sort(inits)


def test_duplicate_name_raises():
    """Duplicate InitStep names are rejected early."""
    inits = [
        InitStep("A", depends_on=[], initializer=lambda: "a1"),
        InitStep("A", depends_on=[], initializer=lambda: "a2"),
    ]
    with pytest.raises(ValueError, match="Duplicate InitStep name"):
        topological_sort(inits)


# --- execute_lazy_inits ---


def test_execute_lazy_inits_order():
    """Execution respects dependency order and passes context."""
    call_order = []
    inits = [
        InitStep(
            "B",
            depends_on=["A"],
            initializer=lambda ctx: (call_order.append("B"), "b")[1],
        ),
        InitStep(
            "A", depends_on=[], initializer=lambda: (call_order.append("A"), "a")[1]
        ),
    ]
    objects = execute_lazy_inits(inits)
    assert call_order == ["A", "B"]
    assert objects == {"A": "a", "B": "b"}


def test_execute_lazy_inits_context_passing():
    """Context dict is passed and grows as execution progresses."""
    inits = [
        InitStep("a", depends_on=[], initializer=lambda: 1),
        InitStep("b", depends_on=["a"], initializer=lambda ctx: ctx["a"] + 1),
        InitStep(
            "c", depends_on=["a", "b"], initializer=lambda ctx: ctx["a"] + ctx["b"]
        ),
    ]
    objects = execute_lazy_inits(inits)
    assert objects == {"a": 1, "b": 2, "c": 3}


# --- build_context_from_objects (parametrized) ---


@pytest.mark.parametrize(
    "objects,check",
    [
        (
            {"fields.U": "v", "fields.p": "p"},
            lambda c: c.fields == {"U": "v", "p": "p"},
        ),
        ({"models.algo": "a"}, lambda c: c.models == {"algo": "a"}),
        ({"operators.mom": "m"}, lambda c: c.models == {"mom": "m"}),
        ({"mesh": "m"}, lambda c: c.mesh == "m"),
        ({"runtime": "r"}, lambda c: c.runTime == "r"),
        ({"custom": "c"}, lambda c: c.models == {"custom": "c"}),
    ],
    ids=["fields", "models", "operators", "mesh", "runtime", "unknown"],
)
def test_build_context_routing(objects, check):
    """build_context_from_objects routes objects to correct Context slots."""
    ctx = build_context_from_objects(objects)
    assert check(ctx)


def test_build_context_mixed():
    """Mixed object types are all routed correctly."""
    objects = {
        "mesh": "mesh_obj",
        "runtime": "runtime_obj",
        "fields.U": "velocity",
        "fields.p": "pressure",
        "models.algo": "algorithm",
        "operators.mom": "momentum",
        "custom": "custom_value",
    }
    ctx = build_context_from_objects(objects)
    assert ctx.mesh == "mesh_obj"
    assert ctx.runTime == "runtime_obj"
    assert ctx.fields == {"U": "velocity", "p": "pressure"}
    assert ctx.models == {
        "algo": "algorithm",
        "mom": "momentum",
        "custom": "custom_value",
    }


# --- execute_initialization ---


def test_execute_initialization():
    """End-to-end: list[InitStep] -> Context."""
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda: "mesh_obj"),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda ctx: f"U_on_{ctx['mesh']}",
        ),
        InitStep("models.algo", depends_on=[], initializer=lambda: "algorithm"),
    ]
    ctx = execute_initialization(inits)
    assert isinstance(ctx, Context)
    assert ctx.mesh == "mesh_obj"
    assert ctx.fields == {"U": "U_on_mesh_obj"}
    assert ctx.models == {"algo": "algorithm"}


def test_execute_initialization_complex():
    """Complex initialization with multi-level dependencies."""
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda: "mesh"),
        InitStep("runtime", depends_on=[], initializer=lambda: "runtime"),
        InitStep("fields.U", depends_on=["mesh"], initializer=lambda: "velocity"),
        InitStep("fields.p", depends_on=["mesh"], initializer=lambda: "pressure"),
        InitStep(
            "operators.mom",
            depends_on=["fields.U", "fields.p"],
            initializer=lambda: "momentum_op",
        ),
        InitStep(
            "models.algo", depends_on=["operators.mom"], initializer=lambda: "algorithm"
        ),
    ]
    ctx = execute_initialization(inits)
    assert ctx.mesh == "mesh"
    assert ctx.runTime == "runtime"
    assert len(ctx.fields) == 2
    assert len(ctx.models) == 2


def test_execute_initialization_raises_structured_graph_error():
    """execute_initialization raises InitializationGraphError with report."""
    inits = [InitStep("A", depends_on=["missing"], initializer=lambda: "a")]

    with pytest.raises(InitializationGraphError) as exc_info:
        execute_initialization(inits)

    report = exc_info.value.report
    assert not report.is_valid
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.code == "missing_dependency"
    assert diag.step_name == "A"
    assert diag.dependency == "missing"


# --- validate_lazy_init_graph ---


def test_validate_graph_missing():
    """Validation detects missing dependencies (exact count, no duplicates)."""
    inits = [
        InitStep("A", depends_on=["missing"], initializer=lambda: "a"),
        InitStep("B", depends_on=["also_missing"], initializer=lambda: "b"),
    ]
    errors = validate_lazy_init_graph(inits)
    assert len(errors) == 2
    assert any("A" in err[0] and "missing" in err[1] for err in errors)
    assert any("B" in err[0] and "also_missing" in err[1] for err in errors)


def test_validate_graph_cycle():
    """Validation detects circular dependencies."""
    inits = [
        InitStep("A", depends_on=["B"], initializer=lambda: "a"),
        InitStep("B", depends_on=["A"], initializer=lambda: "b"),
    ]
    errors = validate_lazy_init_graph(inits)
    assert len(errors) == 1
    assert "Circular" in errors[0][1]


def test_validate_graph_duplicate_name():
    """Validation detects duplicate node names."""
    inits = [
        InitStep("dup", depends_on=[], initializer=lambda: 1),
        InitStep("dup", depends_on=[], initializer=lambda: 2),
    ]
    errors = validate_lazy_init_graph(inits)
    assert len(errors) == 1
    assert "Duplicate InitStep name" in errors[0][1]


def test_validate_graph_clean(linear_chain):
    """Valid graph returns no errors."""
    assert validate_lazy_init_graph(linear_chain) == []


def test_init_graph_validator_structured_cycle_report():
    """InitGraphValidator returns structured cycle diagnostics."""
    inits = [
        InitStep("A", depends_on=["B"], initializer=lambda: "a"),
        InitStep("B", depends_on=["A"], initializer=lambda: "b"),
    ]

    report = InitGraphValidator.validate(inits)
    assert not report.is_valid
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.code == "cycle"
    assert len(diag.cycle) >= 2
    assert "Circular dependency" in diag.message
