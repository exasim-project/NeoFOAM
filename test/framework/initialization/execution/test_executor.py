# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from pathlib import Path

import pytest

from neofoam.framework.initialization.execution.executor import (
    _init_digraph,
    execute_lazy_inits,
    execute_step,
)
from neofoam.framework.initialization.execution.init_result import InitResult
from neofoam.framework.initialization.init_step import (
    InitStep,
    InitStepExecutionError,
)


def test_execute_lazy_inits_order():
    call_order = []
    inits = [
        InitStep(
            "B",
            depends_on=["A"],
            initializer=lambda ctx: (call_order.append("B"), "b")[1],
        ),
        InitStep(
            "A",
            depends_on=[],
            initializer=lambda _ctx: (call_order.append("A"), "a")[1],
        ),
    ]
    results = execute_lazy_inits(inits)
    assert call_order == ["A", "B"]
    assert {r.name: r.value for r in results} == {"A": "a", "B": "b"}


def test_execute_lazy_inits_context_passing():
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


def test_execute_lazy_inits_returns_init_results():
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


def test_execute_step_returns_value():
    step = InitStep("x", initializer=lambda _ctx: 42)

    assert execute_step(step, {}) == 42


def test_execute_step_wraps_unexpected_exceptions():
    def boom(_ctx):
        raise RuntimeError("disk read failed")

    step = InitStep("fields.U", depends_on=["mesh"], initializer=boom)

    with pytest.raises(InitStepExecutionError) as exc_info:
        execute_step(step, {})

    err = exc_info.value
    assert err.step_name == "fields.U"
    assert err.depends_on == ["mesh"]
    assert "disk read failed" in str(err)


def test_execute_step_preserves_value_error():
    step = InitStep(
        "value_err",
        initializer=lambda _ctx: (_ for _ in ()).throw(ValueError("bad value")),
    )
    with pytest.raises(ValueError, match="bad value"):
        execute_step(step, {})


def test_init_digraph_carries_step_attributes_and_edges():
    steps = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: 0),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda _ctx: 1,
            category="fields",
            write=True,
        ),
    ]

    graph = _init_digraph(steps)

    assert graph.has_edge("mesh", "fields.U")
    assert graph.nodes["fields.U"]["category"] == "fields"
    assert graph.nodes["fields.U"]["write"] is True
    assert graph.nodes["fields.U"]["depends_on"] == ["mesh"]


def test_execute_lazy_inits_dumps_dag_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """NEOFOAM_DUMP_INIT_DAG writes the resolved order before executing."""
    target = tmp_path / "init_dag.txt"
    monkeypatch.setenv("NEOFOAM_DUMP_INIT_DAG", str(target))
    inits = [
        InitStep("b", depends_on=["root"], initializer=lambda _ctx: 2),
        InitStep("root", depends_on=[], initializer=lambda _ctx: 0),
    ]

    execute_lazy_inits(inits)

    assert target.is_file()
    body = [
        ln for ln in target.read_text().splitlines() if ln and not ln.startswith("#")
    ]
    # the dependency-free root is executed (and numbered) first
    assert body[0].startswith("   1. root")


def test_execute_lazy_inits_dumps_dot_when_suffix_dot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "init_dag.dot"
    monkeypatch.setenv("NEOFOAM_DUMP_INIT_DAG", str(target))
    execute_lazy_inits([InitStep("root", initializer=lambda _ctx: 0)])

    assert target.read_text().startswith("digraph dag {")


def test_init_step_execution_error_in_pipeline():
    def boom(ctx):
        raise OSError("cannot open file")

    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "m"),
        InitStep("fields.U", depends_on=["mesh"], initializer=boom),
    ]

    with pytest.raises(InitStepExecutionError) as exc_info:
        execute_lazy_inits(inits)

    assert exc_info.value.step_name == "fields.U"
