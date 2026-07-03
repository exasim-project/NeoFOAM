# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the offline trace visualizer (neofoam.telemetry.report).

The reporter reads only the emitted JSON, so these tests build span/summary
records by hand (matching FileSpanExporter's schema) — no OpenTelemetry needed.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from neofoam.telemetry import report

HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def _span(
    name: str,
    span_id: str,
    parent_id: str | None,
    start: str,
    end: str,
    rank: int = 0,
) -> dict:
    """A span record shaped like FileSpanExporter writes."""
    return {
        "name": name,
        "context": {"span_id": span_id},
        "parent_id": parent_id,
        "start_time": start,
        "end_time": end,
        "attributes": {},
        "resource": {"attributes": {"mpi.rank": rank, "mpi.size": 1}},
    }


# A properly nested trace: time_loop > momentum > momentum.solve.
_SPANS = [
    _span(
        "time_loop",
        "0x01",
        None,
        "2026-01-01T00:00:00.000000Z",
        "2026-01-01T00:00:01.000000Z",
    ),
    _span(
        "momentum",
        "0x02",
        "0x01",
        "2026-01-01T00:00:00.100000Z",
        "2026-01-01T00:00:00.400000Z",
    ),
    _span(
        "momentum.solve",
        "0x03",
        "0x02",
        "2026-01-01T00:00:00.200000Z",
        "2026-01-01T00:00:00.300000Z",
    ),
]


def _write_trace_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "telemetry"
    directory.mkdir()
    with open(directory / "rank0.spans.jsonl", "w", encoding="utf-8") as handle:
        for span in _SPANS:
            handle.write(json.dumps(span) + "\n")
    summary = {
        "mpi": {"rank": 0, "size": 1, "par_run": False},
        "spans": {
            "time_loop": {
                "count": 1,
                "total_s": 1.0,
                "mean_s": 1.0,
                "min_s": 1.0,
                "max_s": 1.0,
            },
            "momentum": {
                "count": 1,
                "total_s": 0.3,
                "mean_s": 0.3,
                "min_s": 0.3,
                "max_s": 0.3,
            },
            "momentum.solve": {
                "count": 1,
                "total_s": 0.1,
                "mean_s": 0.1,
                "min_s": 0.1,
                "max_s": 0.1,
            },
        },
    }
    (directory / "rank0.summary.json").write_text(json.dumps(summary))
    return directory


# --- chrome trace export ------------------------------------------------------


def test_to_chrome_trace_normalizes_and_preserves_nesting() -> None:
    trace = report.to_chrome_trace(_SPANS)
    assert trace["displayTimeUnit"] == "ms"

    complete = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    assert {e["name"] for e in complete} == {
        "time_loop",
        "momentum",
        "momentum.solve",
    }

    by_name = {e["name"]: e for e in complete}
    # earliest span (time_loop) is shifted to ts=0; microsecond units
    assert by_name["time_loop"]["ts"] == 0.0
    assert by_name["time_loop"]["dur"] == pytest.approx(1_000_000.0)
    # momentum starts 0.1 s in, lasts 0.3 s
    assert by_name["momentum"]["ts"] == pytest.approx(100_000.0)
    assert by_name["momentum"]["dur"] == pytest.approx(300_000.0)

    # time containment survives (a viewer renders this as a flame stack)
    def contains(outer: str, inner: str) -> bool:
        o, i = by_name[outer], by_name[inner]
        return o["ts"] <= i["ts"] and o["ts"] + o["dur"] >= i["ts"] + i["dur"]

    assert contains("time_loop", "momentum")
    assert contains("momentum", "momentum.solve")


def test_to_chrome_trace_tracks_by_rank() -> None:
    spans = [
        _span(
            "a",
            "0x1",
            None,
            "2026-01-01T00:00:00.000000Z",
            "2026-01-01T00:00:01.000000Z",
            rank=0,
        ),
        _span(
            "b",
            "0x2",
            None,
            "2026-01-01T00:00:00.000000Z",
            "2026-01-01T00:00:01.000000Z",
            rank=3,
        ),
    ]
    trace = report.to_chrome_trace(spans)
    complete = {e["name"]: e for e in trace["traceEvents"] if e["ph"] == "X"}
    assert complete["a"]["pid"] == 0 and complete["a"]["tid"] == 0
    assert complete["b"]["pid"] == 3 and complete["b"]["tid"] == 3
    # a process_name metadata event names each rank track
    names = {
        e["args"]["name"]
        for e in trace["traceEvents"]
        if e["ph"] == "M" and e["name"] == "process_name"
    }
    assert names == {"rank 0", "rank 3"}


def test_to_chrome_trace_empty_is_valid() -> None:
    trace = report.to_chrome_trace([])
    assert trace["traceEvents"] == []
    assert trace["displayTimeUnit"] == "ms"


def test_write_chrome_trace_default_location(tmp_path: Path) -> None:
    directory = _write_trace_dir(tmp_path)
    # resolves the case dir to its telemetry/ sub-dir
    out = report.write_chrome_trace(tmp_path)
    assert out == directory / "trace.json"
    written = json.loads(out.read_text())
    assert len(written["traceEvents"]) == 4  # 3 spans + 1 process_name


def test_load_spans_and_summaries(tmp_path: Path) -> None:
    _write_trace_dir(tmp_path)
    assert len(report.load_spans(tmp_path)) == 3
    summaries = report.load_summaries(tmp_path)
    assert set(summaries) == {0}
    assert summaries[0]["spans"]["time_loop"]["total_s"] == 1.0


def test_resolve_dir_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        report.load_spans(tmp_path)


# --- summary plot -------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="requires matplotlib")
def test_plot_summary_returns_figure_sorted_by_total(tmp_path: Path) -> None:
    directory = _write_trace_dir(tmp_path)
    summary = json.loads((directory / "rank0.summary.json").read_text())
    figure = report.plot_summary(summary)
    ax = figure.axes[0]
    assert len(ax.patches) == 3  # one bar per operation
    # slowest first (invert_yaxis puts it visually on top)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[0] == "time_loop"


@pytest.mark.skipif(not HAS_MPL, reason="requires matplotlib")
def test_plot_summary_top_n(tmp_path: Path) -> None:
    directory = _write_trace_dir(tmp_path)
    summary = json.loads((directory / "rank0.summary.json").read_text())
    figure = report.plot_summary(summary, top=2)
    assert len(figure.axes[0].patches) == 2


@pytest.mark.skipif(not HAS_MPL, reason="requires matplotlib")
def test_write_summary_plot_creates_image(tmp_path: Path) -> None:
    _write_trace_dir(tmp_path)
    out = report.write_summary_plot(tmp_path)
    assert out.name == "summary.png"
    assert out.is_file() and out.stat().st_size > 0
