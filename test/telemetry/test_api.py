# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The telemetry shim: no-op when inactive, real spans when configured."""

import json
import sys
from pathlib import Path

import pytest

from neofoam import telemetry
from neofoam.telemetry import MpiInfo, TelemetryNotInstalledError, TelemetrySettings


def _span_records(case_dir: Path, rank: int = 0) -> list[dict]:
    path = case_dir / "telemetry" / f"rank{rank}.spans.jsonl"
    assert path.is_file(), f"missing span file {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- inactive (default) behaviour -------------------------------------------


def test_inactive_by_default() -> None:
    assert telemetry.is_active() is False


def test_span_is_noop_when_inactive(tmp_path: Path) -> None:
    with telemetry.span("anything", key="value"):
        pass
    assert telemetry.is_active() is False
    assert not (tmp_path / "telemetry").exists()


def test_instrument_passes_through_when_inactive() -> None:
    @telemetry.instrument("named")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_configure_disabled_settings_stays_inactive(tmp_path: Path) -> None:
    telemetry.configure(
        TelemetrySettings(enabled=False), case_dir=tmp_path, mpi=MpiInfo()
    )
    assert telemetry.is_active() is False
    assert not (tmp_path / "telemetry").exists()


def test_shutdown_is_idempotent() -> None:
    telemetry.shutdown()
    telemetry.shutdown()
    assert telemetry.is_active() is False


# --- missing optional dependency ---------------------------------------------


def test_configure_without_opentelemetry_raises_informative_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the extra not being installed: import of opentelemetry fails."""
    for name in list(sys.modules):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)  # type: ignore[arg-type]
    monkeypatch.delitem(sys.modules, "neofoam.telemetry._sdk", raising=False)

    with pytest.raises(TelemetryNotInstalledError, match=r"neofoam\[telemetry\]"):
        telemetry.configure(TelemetrySettings(), case_dir=tmp_path, mpi=MpiInfo())
    assert telemetry.is_active() is False


# --- active behaviour ---------------------------------------------------------

pytest.importorskip("opentelemetry")


def test_configure_activates_and_writes_rank_file(tmp_path: Path) -> None:
    telemetry.configure(
        TelemetrySettings(),
        case_dir=tmp_path,
        mpi=MpiInfo(rank=3, size=8, par_run=True),
    )
    assert telemetry.is_active() is True

    with telemetry.span("outer", stage="demo"):
        with telemetry.span("inner"):
            pass
    telemetry.shutdown()
    assert telemetry.is_active() is False

    records = _span_records(tmp_path, rank=3)
    by_name = {rec["name"]: rec for rec in records}
    assert set(by_name) == {"outer", "inner"}

    # nesting: inner's parent is outer's span id
    assert by_name["inner"]["parent_id"] == by_name["outer"]["context"]["span_id"]
    assert by_name["outer"]["parent_id"] is None
    assert by_name["outer"]["attributes"]["stage"] == "demo"

    # per-processor tagging on the resource
    resource_attrs = by_name["outer"]["resource"]["attributes"]
    assert resource_attrs["mpi.rank"] == 3
    assert resource_attrs["mpi.size"] == 8
    assert resource_attrs["mpi.par_run"] is True
    assert resource_attrs["service.name"] == "neofoam"


def test_span_drops_none_attributes(tmp_path: Path) -> None:
    telemetry.configure(TelemetrySettings(), case_dir=tmp_path, mpi=MpiInfo())
    with telemetry.span("op", kept="yes", dropped=None):
        pass
    telemetry.shutdown()

    (record,) = _span_records(tmp_path)
    assert record["attributes"] == {"kept": "yes"}


def test_instrument_creates_named_span(tmp_path: Path) -> None:
    telemetry.configure(TelemetrySettings(), case_dir=tmp_path, mpi=MpiInfo())

    @telemetry.instrument("my.step")
    def work() -> str:
        return "done"

    @telemetry.instrument()
    def implicit() -> None:
        return None

    assert work() == "done"
    implicit()
    telemetry.shutdown()

    names = {rec["name"] for rec in _span_records(tmp_path)}
    assert "my.step" in names
    assert any("implicit" in name for name in names)


def test_shutdown_writes_summary(tmp_path: Path) -> None:
    telemetry.configure(TelemetrySettings(), case_dir=tmp_path, mpi=MpiInfo())
    for _ in range(3):
        with telemetry.span("repeated"):
            pass
    telemetry.shutdown()

    summary_path = tmp_path / "telemetry" / "rank0.summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    stats = summary["spans"]["repeated"]
    assert stats["count"] == 3
    assert stats["total_s"] >= 0.0
    assert stats["mean_s"] == pytest.approx(stats["total_s"] / 3)


def test_summary_can_be_disabled(tmp_path: Path) -> None:
    telemetry.configure(
        TelemetrySettings(summary=False), case_dir=tmp_path, mpi=MpiInfo()
    )
    with telemetry.span("op"):
        pass
    telemetry.shutdown()

    assert (tmp_path / "telemetry" / "rank0.spans.jsonl").is_file()
    assert not (tmp_path / "telemetry" / "rank0.summary.json").exists()


def test_reconfigure_after_shutdown(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    telemetry.configure(TelemetrySettings(), case_dir=first, mpi=MpiInfo())
    with telemetry.span("run1"):
        pass
    telemetry.shutdown()

    telemetry.configure(TelemetrySettings(), case_dir=second, mpi=MpiInfo())
    with telemetry.span("run2"):
        pass
    telemetry.shutdown()

    assert {rec["name"] for rec in _span_records(first)} == {"run1"}
    assert {rec["name"] for rec in _span_records(second)} == {"run2"}


def test_configure_while_active_replaces_previous(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    telemetry.configure(TelemetrySettings(), case_dir=first, mpi=MpiInfo())
    telemetry.configure(TelemetrySettings(), case_dir=second, mpi=MpiInfo())
    with telemetry.span("later"):
        pass
    telemetry.shutdown()

    # first run was flushed on reconfigure; the new span went to the second dir
    assert {rec["name"] for rec in _span_records(second)} == {"later"}
