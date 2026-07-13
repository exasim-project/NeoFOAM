# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Visualize an emitted telemetry trace — the offline viewer for the run.

Reads only the JSON the :class:`~neofoam.telemetry._sdk.FileSpanExporter`
wrote (``rank<N>.spans.jsonl`` / ``rank<N>.summary.json``), so this module
imports neither OpenTelemetry nor pybFoam: a trace produced on a cluster can
be visualized on a laptop that never installed the ``neofoam[telemetry]``
extra.

Two renderings:

* :func:`write_chrome_trace` — the nested spans as a Chrome Trace-Event
  ``trace.json`` (one process row per MPI rank), opened interactively in
  https://ui.perfetto.dev or ``chrome://tracing`` for a zoomable
  Gantt/flame timeline.
* :func:`plot_summary` / :func:`write_summary_plot` — the per-operation
  aggregate as a horizontal matplotlib bar chart (total wall-clock), for a
  static glance or a report (matplotlib is imported lazily).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_US_PER_S = 1e6

# What FileSpanExporter writes, per rank.
_SPANS_GLOB = "rank*.spans.jsonl"
_SUMMARY_GLOB = "rank*.summary.json"


def _resolve_dir(path: Union[str, Path]) -> Path:
    """Find the directory holding the ``rank*`` telemetry files.

    Accepts either that directory directly or a case directory containing a
    ``telemetry/`` sub-dir (the default output location).
    """
    p = Path(path)
    if any(p.glob(_SPANS_GLOB)) or any(p.glob(_SUMMARY_GLOB)):
        return p
    nested = p / "telemetry"
    if nested.is_dir() and (
        any(nested.glob(_SPANS_GLOB)) or any(nested.glob(_SUMMARY_GLOB))
    ):
        return nested
    raise FileNotFoundError(
        f"no telemetry files ({_SPANS_GLOB} / {_SUMMARY_GLOB}) under {p}"
    )


def load_spans(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read every span record from all ``rank<N>.spans.jsonl`` under *path*."""
    directory = _resolve_dir(path)
    spans: list[dict[str, Any]] = []
    for spans_file in sorted(directory.glob(_SPANS_GLOB)):
        with open(spans_file, encoding="utf-8") as handle:
            spans.extend(json.loads(line) for line in handle if line.strip())
    return spans


def load_summaries(path: Union[str, Path]) -> dict[int, dict[str, Any]]:
    """Read every ``rank<N>.summary.json`` under *path*, keyed by MPI rank."""
    directory = _resolve_dir(path)
    summaries: dict[int, dict[str, Any]] = {}
    for summary_file in sorted(directory.glob(_SUMMARY_GLOB)):
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        rank = int(data.get("mpi", {}).get("rank", 0))
        summaries[rank] = data
    return summaries


def _epoch_us(iso_timestamp: str) -> float:
    """ISO-8601 span timestamp -> microseconds since the epoch (float)."""
    # OpenTelemetry emits e.g. ``2026-07-03T07:27:37.533827Z``; the trailing
    # ``Z`` is only accepted by ``fromisoformat`` from 3.11, so normalise it
    # (the package targets 3.10).
    dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    return dt.timestamp() * _US_PER_S


def _span_rank(span: dict[str, Any]) -> int:
    return int(span.get("resource", {}).get("attributes", {}).get("mpi.rank", 0))


def to_chrome_trace(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert emitted span records to the Chrome Trace-Event format.

    Each span becomes a complete (``ph: "X"``) event on a per-rank track, so
    the natural time-nesting (a parent fully contains its children) renders as
    a flame stack. Timestamps are shifted so the earliest span starts at 0.
    The result is a ``dict`` ready for :func:`json.dump`; open it in
    https://ui.perfetto.dev or ``chrome://tracing``.
    """
    timed = [s for s in spans if s.get("start_time") and s.get("end_time")]
    events: list[dict[str, Any]] = []
    if not timed:
        return {"traceEvents": events, "displayTimeUnit": "ms"}

    starts = {id(s): _epoch_us(s["start_time"]) for s in timed}
    t0 = min(starts.values())

    for rank in sorted({_span_rank(s) for s in timed}):
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": rank,
                "args": {"name": f"rank {rank}"},
            }
        )

    for span in sorted(timed, key=lambda s: starts[id(s)]):
        rank = _span_rank(span)
        ts = starts[id(span)] - t0
        dur = _epoch_us(span["end_time"]) - starts[id(span)]
        args: dict[str, Any] = dict(span.get("attributes", {}))
        args["span_id"] = span.get("context", {}).get("span_id")
        args["parent_id"] = span.get("parent_id")
        events.append(
            {
                "name": span.get("name", "span"),
                "cat": "span",
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": rank,
                "tid": rank,
                "args": args,
            }
        )

    return {"traceEvents": events, "displayTimeUnit": "ms"}


def write_chrome_trace(
    path: Union[str, Path], output: Optional[Union[str, Path]] = None
) -> Path:
    """Write a Chrome Trace-Event ``trace.json`` for the trace under *path*.

    *output* defaults to ``<telemetry-dir>/trace.json``. Returns the path
    written. Open the file in https://ui.perfetto.dev or ``chrome://tracing``.
    """
    directory = _resolve_dir(path)
    out_path = Path(output) if output is not None else directory / "trace.json"
    trace = to_chrome_trace(load_spans(directory))
    out_path.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    return out_path


def _load_summary(summary: Union[str, Path, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(summary, dict):
        return summary
    return cast("dict[str, Any]", json.loads(Path(summary).read_text(encoding="utf-8")))


def plot_summary(
    summary: Union[str, Path, dict[str, Any]],
    *,
    top: Optional[int] = None,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """Horizontal bar chart of per-operation total wall-clock (seconds).

    *summary* is a ``rank<N>.summary.json`` path or its loaded ``dict``. Bars
    are sorted by ``total_s`` descending; *top* keeps only the N slowest. Pass
    an existing *ax* to draw into a caller-owned figure (e.g. so sphinx-gallery
    or a notebook captures it); otherwise a fresh figure is created and
    returned. Requires matplotlib (imported lazily).
    """
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised via the error test
        raise ImportError(
            "plot_summary requires matplotlib: pip install matplotlib "
            "(or install the neofoam[telemetry] extra)"
        ) from exc

    data = _load_summary(summary)
    spans: dict[str, dict[str, float]] = data.get("spans", {})
    ranked = sorted(spans.items(), key=lambda item: item[1]["total_s"], reverse=True)
    if top is not None:
        ranked = ranked[:top]

    names = [name for name, _ in ranked]
    totals = [stats["total_s"] for _, stats in ranked]

    if ax is None:
        figure = Figure(figsize=(8, max(2.0, 0.4 * len(names) + 1)))
        axes = figure.subplots()
    else:
        axes = ax
        figure = cast("Figure", axes.figure)

    positions = range(len(names))
    axes.barh(list(positions), totals, color="#4c78a8")
    axes.set_yticks(list(positions))
    axes.set_yticklabels(names)
    axes.invert_yaxis()  # slowest at the top
    axes.set_xlabel("total wall-clock [s]")
    rank = data.get("mpi", {}).get("rank", 0)
    axes.set_title(f"operation totals (rank {rank})")
    figure.tight_layout()
    return figure


def write_summary_plot(
    path: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    *,
    top: Optional[int] = None,
    rank: Optional[int] = None,
) -> Path:
    """Render a summary bar chart to an image file (PNG by default).

    Picks the lowest rank's summary unless *rank* is given. *output* defaults
    to ``<telemetry-dir>/summary.png``. Returns the path written.
    """
    directory = _resolve_dir(path)
    summaries = load_summaries(directory)
    if not summaries:
        raise FileNotFoundError(f"no {_SUMMARY_GLOB} under {directory}")
    chosen = rank if rank is not None else min(summaries)
    if chosen not in summaries:
        raise KeyError(f"rank {chosen} not among summaries {sorted(summaries)}")

    figure = plot_summary(summaries[chosen], top=top)
    out_path = Path(output) if output is not None else directory / "summary.png"
    figure.savefig(out_path, dpi=150)
    return out_path
