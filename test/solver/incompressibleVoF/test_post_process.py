# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: a VoF case's declared table becomes ``postProcessing/<name>.csv``.

**Integration, real OpenFOAM.** ``test/postprocess`` pins the package and
``test/solver/incompressibleFluid/test_post_process.py`` pins the same seam for
the single-phase solver; this module pins it for ``incompressibleVoF`` — that
the solver loads the case's declaration, steps ``post_process`` last in its own
(solver-owned) time loop, and that the number in the CSV is the volume integral
of the phase fraction as it was written.

**The case.** ``cases/damBreakBox`` overlays ``cases/vofRow4/common`` — the same
interFoam damBreak physical properties and dictionaries — with a 4x4x1 unit box
whose only patches are ``noSlip`` walls and one ``empty`` pair, plus a water
column filling the two left-hand cell columns. It is the smallest VoF case that
both *moves* and *conserves*: ``cases/vofRow4/common`` is a through-flow (water
enters at the inlet) and its closed sibling
``models/pressure_velocity/cases/closed`` is one cell tall, where a horizontal
density step is exactly hydrostatic and nothing ever moves. Here the column
collapses to the right under gravity, while every boundary flux is zero — so
``volIntegrate(alpha.water)`` is a genuine mass-conservation check on a field
that is genuinely changing.

**Tolerances.** Two different budgets:

* the *drift* over the run is bounded by the alpha equation's own linear-solver
  tolerance, ``1e-8`` in ``system/fvSolution``'s ``"alpha.water.*"`` dict, once
  per step for five steps — hence ``_CONSERVATION_RTOL`` of 1e-7. Measured drift
  on this case is 5.7e-10, two decades inside it.
* the *last row against a recomputation from disk* is limited only by the ASCII
  round-trip of the written field; ``cases/vofRow4/common`` writes
  ``writePrecision 17``, so ``_READBACK_RTOL`` of 1e-12 is loose by orders of
  magnitude and only covers the summation order.

The cell volume is a pre-written literal: a unit box cut into 4x4x1 cells, so
every cell holds exactly 1/16.

**Subprocess per run.** One ``Foam::Time`` per process, as everywhere else in
these tests, so the solver runs in a fresh interpreter with ``FOAM_SIGFPE`` off.
The reference read of ``alpha.water`` gets its own process again via
``CaseDir.read_field``.

**Placement.** That ``post_process`` is stepped *after* ``write_output`` is what
the CSV rows mean, and a solved case cannot show it (nothing mutates the fields
between the two). It is asserted on the built graph instead, case-free, with the
core models the graph step reads faked down to their names and op names.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
from numpy.testing import assert_allclose

from neofoam.framework.context import Context
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata
from neofoam.solver.incompressibleVoF.incompressibleVoF import execution_graph
from neofoam.tooling.casebuild import CaseDir, block_mesh, patch

from ..incompressibleFluid.solved_case import read_table
from .conftest import build_case, overlay

_CASES = Path(__file__).parent / "cases"

#: ``cases/damBreakBox``: a unit box cut into 4x4x1 cells.
_CELL_VOLUME = 1.0 / 16.0

#: Eight of the sixteen cells start full of water.
_INITIAL_WATER_VOLUME = 8 * _CELL_VOLUME

#: Five steps of 0.005 s: enough for the column to collapse across two cells.
_END_TIME = 0.025
_DELTA_T = 0.005

#: See the module docstring: the alpha solver's own 1e-8, once per step.
_CONSERVATION_RTOL = 1e-7

#: See the module docstring: ASCII round-trip of ``writePrecision 17``.
_READBACK_RTOL = 1e-12

_SOLVER_DRIVER = """
import os
os.environ["FOAM_SIGFPE"] = ""
from neofoam.solver.incompressibleVoF import run
run(["incompressibleVoF"])
"""


def solved_box(dest: Path, *, declarations_dir: Optional[Path] = None) -> CaseDir:
    """Build ``cases/damBreakBox`` at *dest*, overlay *declarations_dir*, solve it.

    Pass the directory holding the case's ``system/postProcess.yaml`` as
    *declarations_dir*, or leave it out for a case that declares no table.
    """
    layers = [_CASES / "damBreakBox"]
    if declarations_dir is not None:
        layers.append(declarations_dir)
    case = CaseDir(
        build_case(
            dest,
            overlay(*layers),
            patch("system/controlDict", endTime=_END_TIME, deltaT=_DELTA_T),
            block_mesh(),
        )
    )

    solve = subprocess.run(
        [sys.executable, "-c", _SOLVER_DRIVER],
        cwd=case.path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert solve.returncode == 0, (
        f"incompressibleVoF aborted on {dest.name}:\n{solve.stdout[-2000:]}\n{solve.stderr[-2000:]}"
    )
    return case


@pytest.fixture(scope="module")
def box_with_a_table(tmp_path_factory: pytest.TempPathFactory) -> CaseDir:
    """The collapsing water column solved for five steps, with one table on it."""
    dest = tmp_path_factory.mktemp("postprocess_yaml") / "damBreakBox"
    return solved_box(dest, declarations_dir=_CASES / "postprocess_yaml")


def test_a_declared_table_gets_one_row_per_time_step(box_with_a_table: CaseDir) -> None:
    header, rows = read_table(box_with_a_table, "water_volume")

    assert header == ["time", "water_volume"]
    assert [float(row[0]) for row in rows] == pytest.approx([0.005, 0.010, 0.015, 0.020, 0.025])


def test_the_water_volume_is_conserved_over_the_run(box_with_a_table: CaseDir) -> None:
    _, rows = read_table(box_with_a_table, "water_volume")

    assert_allclose(
        [float(row[1]) for row in rows],
        _INITIAL_WATER_VOLUME,
        rtol=_CONSERVATION_RTOL,
        err_msg="damBreakBox: a closed box must not gain or lose water",
    )


def test_the_last_row_is_the_volume_integral_of_the_field_as_written(
    box_with_a_table: CaseDir,
) -> None:
    written_alpha = box_with_a_table.read_field("alpha.water", time=str(_END_TIME))

    _, rows = read_table(box_with_a_table, "water_volume")

    assert_allclose(
        float(rows[-1][1]),
        float(written_alpha.sum()) * _CELL_VOLUME,
        rtol=_READBACK_RTOL,
        err_msg=f"damBreakBox at t={_END_TIME}",
    )


def test_a_case_declaring_no_table_writes_no_output_directory(tmp_path: Path) -> None:
    case = solved_box(tmp_path / "damBreakBox")

    assert not (case.path / "postProcessing").exists()


def _operation(name: str) -> Operation:
    return Operation(func=SequentialOp(lambda _ctx: None), metadata=OperationMetadata(op_name=name))


def _core_spec(name: str, op_names: list[str]) -> SimpleNamespace:
    """A bare core-model spec as the graph step reads it: a name and named ops."""
    operations = [_operation(op_name) for op_name in op_names]
    return SimpleNamespace(name=name, build_operations_for=lambda _self: operations)


def _core_runtime(spec_name: str, op_names: list[str]) -> SimpleNamespace:
    """An instantiated core model as the graph step reads it (``ModelRuntime``)."""
    return SimpleNamespace(
        spec=SimpleNamespace(name=spec_name),
        name=f"{spec_name}_main",
        operations=[_operation(op_name) for op_name in op_names],
    )


def test_the_time_loop_ends_with_write_output_then_post_process() -> None:
    state = SimpleNamespace(
        core_models=[
            _core_spec("MULES", ["alpha_advection"]),
            _core_spec("Pimple", ["inner_loop", "mesh_update", "momentum", "continuity"]),
            _core_runtime("postProcess", ["post_process"]),
        ],
        optional_models=[],
    )
    solver = SimpleNamespace(
        state=state,
        operations=Operations(
            [
                _operation(name)
                for name in (
                    "set_time_step",
                    "increment_time",
                    "turbulence_correction",
                    "write_output",
                )
            ]
        ),
    )

    builder, _ = execution_graph(solver, Context(fields={}, models={}))

    time_loop = builder.operations["time_loop"]
    assert [op.operation_name for op in time_loop.sub_operations][-2:] == [
        "write_output",
        "post_process",
    ]
