# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""FileSpanExporter unit tests (requires the optional opentelemetry extra)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402

from neofoam.telemetry import MpiInfo  # noqa: E402
from neofoam.telemetry._sdk import FileSpanExporter  # noqa: E402


def _exporter(
    tmp_path: Path, mpi: MpiInfo | None = None, write_summary: bool = True
) -> FileSpanExporter:
    resolved = mpi if mpi is not None else MpiInfo()
    return FileSpanExporter(
        directory=tmp_path, mpi_source=lambda: resolved, write_summary=write_summary
    )


def _provider_with(exporter: FileSpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def test_exporter_writes_one_json_line_per_span(tmp_path: Path) -> None:
    provider = _provider_with(_exporter(tmp_path))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("a"):
        pass
    with tracer.start_as_current_span("b"):
        pass
    provider.shutdown()

    lines = (tmp_path / "rank0.spans.jsonl").read_text().splitlines()
    assert [json.loads(line)["name"] for line in lines] == ["a", "b"]


def test_exporter_rank_names_files(tmp_path: Path) -> None:
    provider = _provider_with(_exporter(tmp_path, MpiInfo(rank=5, size=8)))
    with provider.get_tracer("test").start_as_current_span("a"):
        pass
    provider.shutdown()

    assert (tmp_path / "rank5.spans.jsonl").is_file()
    assert (tmp_path / "rank5.summary.json").is_file()


def test_exporter_stamps_mpi_attributes_into_records(tmp_path: Path) -> None:
    provider = _provider_with(
        _exporter(tmp_path, MpiInfo(rank=1, size=4, par_run=True))
    )
    with provider.get_tracer("test").start_as_current_span("a"):
        pass
    provider.shutdown()

    (line,) = (tmp_path / "rank1.spans.jsonl").read_text().splitlines()
    attrs = json.loads(line)["resource"]["attributes"]
    assert attrs["mpi.rank"] == 1
    assert attrs["mpi.size"] == 4
    assert attrs["mpi.par_run"] is True


def test_mpi_resolved_lazily_at_first_export(tmp_path: Path) -> None:
    """The rank may only be known after configure (argList initializes MPI)."""
    box = {"rank": 0}
    exporter = FileSpanExporter(
        directory=tmp_path,
        mpi_source=lambda: MpiInfo(rank=box["rank"], size=4, par_run=True),
        write_summary=True,
    )
    provider = _provider_with(exporter)
    box["rank"] = 2  # MPI comes up after the exporter was built

    with provider.get_tracer("test").start_as_current_span("a"):
        pass
    provider.shutdown()

    assert (tmp_path / "rank2.spans.jsonl").is_file()
    summary = json.loads((tmp_path / "rank2.summary.json").read_text())
    assert summary["mpi"] == {"rank": 2, "size": 4, "par_run": True}


def test_summary_aggregation_math(tmp_path: Path) -> None:
    provider = _provider_with(_exporter(tmp_path))
    tracer = provider.get_tracer("test")

    for _ in range(4):
        with tracer.start_as_current_span("hot"):
            pass
    with tracer.start_as_current_span("cold"):
        pass
    provider.shutdown()

    summary = json.loads((tmp_path / "rank0.summary.json").read_text())
    assert summary["mpi"]["rank"] == 0

    hot = summary["spans"]["hot"]
    assert hot["count"] == 4
    assert hot["min_s"] <= hot["mean_s"] <= hot["max_s"]
    assert hot["mean_s"] == pytest.approx(hot["total_s"] / 4)
    assert summary["spans"]["cold"]["count"] == 1


def test_no_summary_file_when_disabled(tmp_path: Path) -> None:
    provider = _provider_with(_exporter(tmp_path, write_summary=False))
    with provider.get_tracer("test").start_as_current_span("a"):
        pass
    provider.shutdown()

    assert not (tmp_path / "rank0.summary.json").exists()
