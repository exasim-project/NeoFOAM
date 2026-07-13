# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The only module that imports OpenTelemetry.

Loaded lazily by :func:`neofoam.telemetry.configure` so the rest of the
package (and everything importing the shim) works without the optional
``neofoam[telemetry]`` extra installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, IO, Optional, Sequence

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Tracer

from .settings import MpiInfo, TelemetrySettings

_NS_PER_S = 1e9


class FileSpanExporter(SpanExporter):
    """Write one JSON line per finished span to ``rank<N>.spans.jsonl``.

    Synchronous (used with :class:`SimpleSpanProcessor`) so no background
    thread exists — safe under MPI. On ``shutdown`` an aggregated
    per-span-name timing summary is written to ``rank<N>.summary.json``.

    ``mpi_source`` is resolved lazily at the first export: telemetry is
    configured before the solver's ``argList`` initializes MPI, so the rank
    is only reliable once the first span (the ``argList``/``Time`` init
    step) has finished. The resolved rank names the files and its
    attributes are stamped into every record's resource.
    """

    def __init__(
        self,
        directory: Path,
        mpi_source: Callable[[], MpiInfo],
        write_summary: bool,
    ) -> None:
        self._directory = Path(directory)
        self._mpi_source = mpi_source
        self._write_summary = write_summary
        self._file: Optional[IO[str]] = None
        self._mpi = MpiInfo()
        self._closed = False
        self._stats: dict[str, dict[str, float]] = {}

    def _ensure_open(self) -> IO[str]:
        if self._file is None:
            self._mpi = self._mpi_source()
            self._directory.mkdir(parents=True, exist_ok=True)
            self._file = open(
                self._directory / f"rank{self._mpi.rank}.spans.jsonl",
                "w",
                encoding="utf-8",
            )
        return self._file

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._closed:
            return SpanExportResult.FAILURE
        file = self._ensure_open()
        for span in spans:
            record = json.loads(span.to_json())
            record.setdefault("resource", {}).setdefault("attributes", {}).update(
                {
                    "mpi.rank": self._mpi.rank,
                    "mpi.size": self._mpi.size,
                    "mpi.par_run": self._mpi.par_run,
                }
            )
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._record_stats(span)
        file.flush()
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if self._file is not None:
            self._file.flush()
        return True

    def shutdown(self) -> None:
        if self._closed:
            return
        self._ensure_open().close()
        self._closed = True
        if self._write_summary:
            self._write_summary_file()

    def _record_stats(self, span: ReadableSpan) -> None:
        if span.start_time is None or span.end_time is None:
            return
        duration_s = (span.end_time - span.start_time) / _NS_PER_S
        stats = self._stats.setdefault(
            span.name,
            {"count": 0, "total_s": 0.0, "min_s": duration_s, "max_s": duration_s},
        )
        stats["count"] += 1
        stats["total_s"] += duration_s
        stats["min_s"] = min(stats["min_s"], duration_s)
        stats["max_s"] = max(stats["max_s"], duration_s)

    def _write_summary_file(self) -> None:
        spans: dict[str, dict[str, float]] = {}
        for name, stats in sorted(self._stats.items()):
            spans[name] = {
                "count": int(stats["count"]),
                "total_s": stats["total_s"],
                "mean_s": stats["total_s"] / stats["count"],
                "min_s": stats["min_s"],
                "max_s": stats["max_s"],
            }
        summary = {
            "mpi": {
                "rank": self._mpi.rank,
                "size": self._mpi.size,
                "par_run": self._mpi.par_run,
            },
            "spans": spans,
        }
        summary_path = self._directory / f"rank{self._mpi.rank}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


@dataclass
class ActiveTelemetry:
    """Live tracing state held by the shim while telemetry is configured."""

    provider: TracerProvider
    tracer: Tracer
    exporter: FileSpanExporter

    def shutdown(self) -> None:
        self.provider.shutdown()


def _lazy_mpi_source() -> MpiInfo:
    from .mpi import current_mpi_info

    return current_mpi_info()


def _fixed_mpi_source(mpi: MpiInfo) -> Callable[[], MpiInfo]:
    def source() -> MpiInfo:
        return mpi

    return source


def start(
    settings: TelemetrySettings, case_dir: Path, mpi: Optional[MpiInfo]
) -> ActiveTelemetry:
    """Build a per-run tracer (kept out of the global otel provider).

    ``mpi=None`` defers rank resolution to the exporter's first export —
    by then the solver's ``argList`` has initialized MPI.
    """
    mpi_source = _lazy_mpi_source if mpi is None else _fixed_mpi_source(mpi)
    exporter = FileSpanExporter(
        directory=case_dir / settings.directory,
        mpi_source=mpi_source,
        write_summary=settings.summary,
    )
    attributes: dict[str, Any] = {"service.name": settings.service_name}
    provider = TracerProvider(resource=Resource.create(attributes))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return ActiveTelemetry(
        provider=provider, tracer=provider.get_tracer("neofoam"), exporter=exporter
    )
