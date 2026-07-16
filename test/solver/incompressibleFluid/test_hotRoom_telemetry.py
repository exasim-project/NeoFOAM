# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Integration: OpenTelemetry tracing on the hotRoom incompressibleFluid case.

Runs the bundled ``tutorials/hotRoom`` for two time steps with the
``telemetry`` controlDict dict enabled and asserts the per-rank span file
and timing summary describe the whole run: init steps, the time loop, the
PIMPLE inner loop, and per-processor (rank) tagging.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("opentelemetry")

from neofoam.solver.incompressibleFluid import run  # noqa: E402

from .comparison_helpers import setup_case  # noqa: E402

# Match the hotRoom comparison test: intermediate deltas legitimately
# underflow on the first iteration of an in-process run.
os.environ["FOAM_SIGFPE"] = ""

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_HOTROOM = _REPO_ROOT / "tutorials" / "hotRoom"

_TELEMETRY_DICT = """
telemetry
{
    enabled     yes;
}
"""


def _prepare_case(target: Path, end_time: float = 4.0) -> Path:
    setup_case(_HOTROOM, target, end_time, write_interval=2.0, run_setfields=True)
    control_dict = target / "system" / "controlDict"
    control_dict.write_text(control_dict.read_text() + _TELEMETRY_DICT)
    return target


def _run_case(case: Path) -> None:
    cwd = Path.cwd()
    os.chdir(case)
    try:
        run(["incompressibleFluid"], log_file=case / "solver.log")
    finally:
        os.chdir(cwd)


def _span_records(case: Path, rank: int = 0) -> list[dict[str, Any]]:
    path = case / "telemetry" / f"rank{rank}.spans.jsonl"
    assert path.is_file(), f"missing span file {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parent_chain(
    record: dict[str, Any], by_id: dict[str, dict[str, Any]]
) -> list[str]:
    names = []
    parent = record["parent_id"]
    while parent is not None:
        names.append(by_id[parent]["name"])
        parent = by_id[parent]["parent_id"]
    return names


def test_hotRoom_telemetry_end_to_end(tmp_path: Path) -> None:
    case = _prepare_case(tmp_path / "hotRoom_telemetry")
    _run_case(case)

    records = _span_records(case)
    names = {r["name"] for r in records}
    by_id = {r["context"]["span_id"]: r for r in records}

    # the whole hierarchy is present: root, init phase, loop, inner loop ops
    assert "solver.run" in names
    assert "initialization" in names
    assert any(name.startswith("init.") for name in names)
    for op_name in (
        "time_loop",
        "set_time_step",
        "increment_time",
        "inner_loop",
        "momentum",
        "continuity",
        "write_output",
    ):
        assert op_name in names, f"operation span '{op_name}' missing"

    # the root span has no parent; init steps sit under initialization < solver.run
    (root,) = (r for r in records if r["name"] == "solver.run")
    assert root["parent_id"] is None
    init_step = next(r for r in records if r["name"].startswith("init."))
    assert _parent_chain(init_step, by_id) == ["initialization", "solver.run"]

    # momentum spans nest inner_loop < time_loop < solver.run, one per iteration
    momentum = [r for r in records if r["name"] == "momentum"]
    assert len(momentum) >= 2  # at least one inner iteration per time step
    assert _parent_chain(momentum[0], by_id) == [
        "inner_loop",
        "time_loop",
        "solver.run",
    ]

    # per-processor tagging: serial run is rank 0 of 1
    resource_attrs = root["resource"]["attributes"]
    assert resource_attrs["mpi.rank"] == 0
    assert resource_attrs["mpi.size"] == 1
    assert resource_attrs["mpi.par_run"] is False
    assert resource_attrs["service.name"] == "incompressibleFluid"


def test_hotRoom_telemetry_summary_counts_match_spans(tmp_path: Path) -> None:
    case = _prepare_case(tmp_path / "hotRoom_summary")
    _run_case(case)

    summary_path = case / "telemetry" / "rank0.summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["mpi"]["rank"] == 0

    records = _span_records(case)
    momentum_count = sum(1 for r in records if r["name"] == "momentum")
    stats = summary["spans"]["momentum"]
    assert stats["count"] == momentum_count
    assert stats["total_s"] > 0.0
    assert stats["mean_s"] == pytest.approx(stats["total_s"] / stats["count"])
    assert summary["spans"]["solver.run"]["count"] == 1


def test_hotRoom_without_telemetry_dict_writes_nothing(tmp_path: Path) -> None:
    case = tmp_path / "hotRoom_no_telemetry"
    setup_case(_HOTROOM, case, 4.0, write_interval=2.0, run_setfields=True)
    _run_case(case)

    assert not (case / "telemetry").exists()


def test_second_in_process_run_reconfigures_cleanly(tmp_path: Path) -> None:
    first = _prepare_case(tmp_path / "hotRoom_first")
    second = _prepare_case(tmp_path / "hotRoom_second")

    _run_case(first)
    _run_case(second)

    for case in (first, second):
        names = {r["name"] for r in _span_records(case)}
        assert "solver.run" in names
        assert "momentum" in names


# --- MPI ------------------------------------------------------------------------

# The two-subdomain decomposeParDict and the ``-parallel`` driver live in real
# files (shared with test_pitzDaily_comparison) rather than inline strings.
_DECOMPOSE_PAR_DICT = Path(__file__).parent / "_parallel_decomposeParDict"
_PARALLEL_DRIVER = Path(__file__).parent / "_parallel_driver.py"


def _mpi_available() -> bool:
    return (
        shutil.which("mpirun") is not None and shutil.which("decomposePar") is not None
    )


@pytest.mark.skipif(not _mpi_available(), reason="mpirun/decomposePar not available")
def test_hotRoom_telemetry_parallel_writes_one_file_per_rank(tmp_path: Path) -> None:
    case = _prepare_case(tmp_path / "hotRoom_parallel")
    shutil.copyfile(_DECOMPOSE_PAR_DICT, case / "system" / "decomposeParDict")

    result = subprocess.run(
        ["decomposePar", "-case", str(case)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"decomposePar failed: {result.stderr}"

    import sys

    result = subprocess.run(
        ["mpirun", "-np", "2", sys.executable, str(_PARALLEL_DRIVER)],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
    )
    assert result.returncode == 0, (
        f"parallel run failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    for rank in (0, 1):
        records = _span_records(case, rank=rank)
        (root,) = (r for r in records if r["name"] == "solver.run")
        resource_attrs = root["resource"]["attributes"]
        assert resource_attrs["mpi.rank"] == rank
        assert resource_attrs["mpi.size"] == 2
        assert resource_attrs["mpi.par_run"] is True
        assert any(r["name"] == "momentum" for r in records)
