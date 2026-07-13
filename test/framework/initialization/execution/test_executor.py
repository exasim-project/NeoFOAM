# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for executing lazy init steps in dependency order."""

import importlib.util
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from neofoam import telemetry
from neofoam.framework.initialization.execution.executor import (
    execute_lazy_inits,
    execute_step,
)
from neofoam.framework.initialization.execution.init_result import InitResult
from neofoam.framework.initialization.init_step import (
    InitStep,
    InitStepExecutionError,
)
from neofoam.telemetry import MpiInfo, TelemetrySettings

HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None


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


# --- telemetry instrumentation ------------------------------------------------


@pytest.fixture
def traced(tmp_path: Path) -> Iterator[Path]:
    """Activate telemetry for one test; yields the case dir."""
    telemetry.configure(TelemetrySettings(), case_dir=tmp_path, mpi=MpiInfo())
    yield tmp_path
    telemetry.shutdown()


def _span_records(case_dir: Path) -> list[dict[str, Any]]:
    telemetry.shutdown()  # flush
    path = case_dir / "telemetry" / "rank0.spans.jsonl"
    assert path.is_file(), f"missing span file {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.skipif(not HAS_OTEL, reason="requires the neofoam[telemetry] extra")
def test_execute_lazy_inits_emits_init_spans(traced: Path) -> None:
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "m"),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda ctx: ctx["mesh"] + "!",
            category="fields",
        ),
    ]
    results = execute_lazy_inits(inits)
    assert {r.name: r.value for r in results} == {"mesh": "m", "fields.U": "m!"}

    records = {r["name"]: r for r in _span_records(traced)}
    root = records["initialization"]
    assert records["init.mesh"]["parent_id"] == root["context"]["span_id"]
    assert records["init.fields.U"]["parent_id"] == root["context"]["span_id"]
    assert records["init.fields.U"]["attributes"]["category"] == "fields"


def test_execute_lazy_inits_without_telemetry_unchanged(tmp_path: Path) -> None:
    inits = [InitStep("a", depends_on=[], initializer=lambda _ctx: 1)]
    results = execute_lazy_inits(inits)
    assert results[0].value == 1
    assert not (tmp_path / "telemetry").exists()
