# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: a NeoN case's declared tables become ``postProcessing/<name>.csv``.

**Integration, real NeoN + OpenFOAM.** The unit tests under ``test/postprocess``
pin the package; this module pins the *NeoN* seam — that ``incompressibleFluidNeoN``
loads the case's declaration, that the ``internal`` source reads a NeoN volume
field (a host copy of its internal vector, not ``internalField()``), that the
cell geometry comes off the adapter's ``MeshAdapter`` on ``Context.mesh``, and
that the number in the CSV is the volume integral of the field as it was written.

**The case.** ``test/setup_pimple`` under the ``casebuild`` pipeline, the same
cavity ``test_cavity_run.py`` runs: 20x20x1 uniform cells in a 0.1 m box, so
every cell carries the same volume. Three steps of 0.005 s with
``writeInterval 1`` so the last time directory and the last CSV row describe the
same state. NeoN/Kokkos and OpenFOAM keep per-process global state, so the
solver runs in a fresh interpreter.

**Tolerance.** The reference value is recomputed from the *written* time
directory, so its error budget is the ASCII round-trip of those files — the case
writes ``writePrecision 15``, and ``_RTOL`` leaves several decades of headroom
over the sum of 400 cell values.

**Loop order.** That ``post_process`` is stepped *after* ``write_output`` is what
a CSV row would mean, and it is asserted on the built graph with the core models
the graph step reads faked down to their spec name and op names — case-free.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.model import ModelRuntime
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata
from neofoam.solver.incompressibleFluidNeoN import config_classes
from neofoam.solver.incompressibleFluidNeoN.incompressibleFluidNeoN import execution_graph
from neofoam.tooling.casebuild import CaseDir, block_mesh, from_template, patch
from solver.incompressibleFluid.solved_case import read_table

_CASES = Path(__file__).parent / "cases"

#: ``test/setup_pimple/system/blockMeshDict``: 20x20x1 cells, ``scale 0.1`` on a
#: unit box, so every cell is (0.1/20) x (0.1/20) x 0.1 m.
_CELL_VOLUME = 0.1 / 20 * 0.1 / 20 * 0.1

#: Three steps of the case's ``deltaT 0.005``.
_END_TIME = 0.015

#: See the module docstring: ASCII round-trip of ``writePrecision 15``.
_RTOL = 1e-10


def _overlay(source: Path) -> Any:
    """Copy a checked-in declaration directory over the built case."""

    def step(case: CaseDir) -> None:
        shutil.copytree(source, case.path, dirs_exist_ok=True)

    return step


@pytest.fixture(scope="module")
def cavity_with_tables(tmp_path_factory: pytest.TempPathFactory) -> CaseDir:
    """The NeoN cavity solved for three steps with ``cases/postprocess_yaml`` on it."""
    case = (
        from_template(Path(__file__).parents[2] / "setup_pimple")
        | patch("system/controlDict", endTime=_END_TIME, writeInterval=1)
        | _overlay(_CASES / "postprocess_yaml")
        | block_mesh()
    ).build_at(tmp_path_factory.mktemp("postprocess") / "cavity")

    solve = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neofoam.solver.incompressibleFluidNeoN import run;"
            " run(['incompressibleFluidNeoN'])",
        ],
        cwd=str(case.path),
        env={**os.environ, "FOAM_SIGFPE": "false"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert solve.returncode == 0, (
        f"incompressibleFluidNeoN aborted on {case.path.name}:\n"
        f"{solve.stdout[-2000:]}\n{solve.stderr[-2000:]}"
    )
    return case


def test_a_declared_table_gets_one_row_per_time_step(cavity_with_tables: CaseDir) -> None:
    header, rows = read_table(cavity_with_tables, "volume_p")

    assert header == ["time", "volume_p"]
    assert [float(row[0]) for row in rows] == pytest.approx([0.005, 0.01, 0.015])


def test_the_scalar_row_is_the_volume_integral_of_p_as_written(
    cavity_with_tables: CaseDir,
) -> None:
    written_p = cavity_with_tables.read_field("p", time=str(_END_TIME))

    _, rows = read_table(cavity_with_tables, "volume_p")

    assert float(rows[-1][1]) == pytest.approx(float(written_p.sum()) * _CELL_VOLUME, rel=_RTOL), (
        "volume_p at the last step is not the volume integral of the written p"
    )


def test_the_vector_row_is_the_volume_integral_of_mag_U_as_written(
    cavity_with_tables: CaseDir,
) -> None:
    written_u = cavity_with_tables.read_field("U", time=str(_END_TIME))

    _, rows = read_table(cavity_with_tables, "volume_mag_U")

    assert float(rows[-1][1]) == pytest.approx(
        float(np.linalg.norm(written_u, axis=1).sum()) * _CELL_VOLUME, rel=_RTOL
    ), "volume_mag_U at the last step is not the volume integral of the written |U|"


def test_post_process_config_is_part_of_the_solver_schema() -> None:
    assert "PostProcessConfig" in {c.__name__ for c in config_classes()}


def _operations(op_names: list[str]) -> list[Operation]:
    return [
        Operation(func=SequentialOp(lambda _ctx: None), metadata=OperationMetadata(op_name=name))
        for name in op_names
    ]


def _core_model(spec_name: str, op_names: list[str]) -> ModelRuntime:
    """A built core model as the graph step reads it: a spec name and named ops."""
    operations = _operations(op_names)
    return ModelRuntime(
        spec=SimpleNamespace(name=spec_name, _build_operations_for=lambda _rt: operations),
        name=spec_name,
        config=None,
    )


def _algorithm_model(op_names: list[str]) -> Any:
    """The pressure-velocity spec, which the graph step uses as its own runtime."""
    operations = _operations(op_names)
    return SimpleNamespace(
        spec=SimpleNamespace(name="pimpleNeoN"),
        _build_operations_for=lambda _self: operations,
    )


def test_the_time_loop_ends_with_write_output_then_post_process() -> None:
    state = SimpleNamespace(
        core_models=[
            _algorithm_model(
                ["rotate_and_report", "inner_loop", "momentum", "continuity", "turbulence_correct"]
            ),
            _core_model("solutionLoop", ["set_time_step", "increment_time"]),
            _core_model("fieldWriter", ["write_output"]),
            _core_model("postProcess", ["post_process"]),
        ],
        optional_models=[],
    )

    builder, _ = execution_graph(SimpleNamespace(state=state), Context(fields={}, models={}))

    time_loop = builder.operations["time_loop"]
    assert [op.operation_name for op in time_loop.sub_operations][-2:] == [
        "write_output",
        "post_process",
    ]
