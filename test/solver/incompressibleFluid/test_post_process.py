# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: a case's declared tables become ``postProcessing/<name>.csv``.

**Integration, real OpenFOAM.** The unit tests under ``test/postprocess`` pin the
package; this module pins the *seam* — that ``incompressibleFluid`` loads the
case's declaration, steps ``post_process`` last in the time loop, and that the
number in the CSV is the volume integral of the field as it was written.

Both front doors declare the same table (``cases/postprocess_yaml`` and
``cases/postprocess_script``), so covering the second one is a ``parametrize``
entry, not a second test body.

**The solved case.** Building the cavity, overlaying the declaration and solving
it in a fresh interpreter is :func:`solved_case.solved_cavity`, shared with the
``test/postprocess`` e2e modules; the reference read of ``p`` gets its own
process again via ``CaseDir.read_field``.

**Parallel.** A decomposed run of the same case is
``test_post_process_parallel.py``; nothing here needs MPI.

**Tolerance.** The reference value is recomputed from the *written* ``0.003/p``,
so its error budget is the ASCII round-trip of that file — the case writes
``writePrecision 12``, leaving ~1e-12 relative, and ``_RTOL`` keeps two decades
of headroom over the sum of nine cell values.

**Cadence.** ``cases/postprocess_runtime`` declares a ``runTime`` policy of
0.002 on a ``deltaT`` of 0.001, and ``RunTimeWriteControl`` fires when the
current time has advanced a full interval past the last write — so on a six-step
run the rows land at 0.002, 0.004 and 0.006, and never at an odd step.

**Placement.** That ``post_process`` is stepped *after* ``write_output`` is what
the CSV rows mean, and a solved case cannot show it (nothing mutates the fields
between the two). It is asserted on the built graph instead, case-free, with the
core models the graph step reads faked down to their spec name and op names.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from neofoam.framework.context import Context
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata
from neofoam.solver.incompressibleFluid.incompressibleFluid import execution_graph
from neofoam.tooling.casebuild import CaseDir

from .solved_case import read_table, solved_cavity

_CASES = Path(__file__).parent / "cases"

#: ``cases/cavity3x3/system/blockMeshDict``: 3x3x1 uniform cells in a
#: 0.1 x 0.1 x 0.01 m box, so every cell carries the same volume.
_CELL_VOLUME = 1e-4 / 9

#: See the module docstring: ASCII round-trip of ``writePrecision 12``.
_RTOL = 1e-10


@pytest.fixture(
    scope="module",
    params=["postprocess_yaml", "postprocess_script"],
    ids=["declared_in_yaml", "declared_in_a_script"],
)
def cavity_with_a_table(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> CaseDir:
    """The cavity solved for three steps with one ``volume_p`` table on it."""
    dest = tmp_path_factory.mktemp(request.param) / "cavity"
    return solved_cavity(dest, declarations_dir=_CASES / str(request.param))


def test_a_declared_table_gets_one_row_per_time_step(cavity_with_a_table: CaseDir) -> None:
    header, rows = read_table(cavity_with_a_table, "volume_p")

    assert header == ["time", "volume_p"]
    assert [float(row[0]) for row in rows] == pytest.approx([0.001, 0.002, 0.003])


def test_the_row_is_the_volume_integral_of_the_field_as_written(
    cavity_with_a_table: CaseDir,
) -> None:
    written_p = cavity_with_a_table.read_field("p", time="0.003")

    _, rows = read_table(cavity_with_a_table, "volume_p")

    assert float(rows[-1][1]) == pytest.approx(float(written_p.sum()) * _CELL_VOLUME, rel=_RTOL)


def test_a_run_time_cadence_writes_only_on_its_own_interval(tmp_path: Path) -> None:
    case = solved_cavity(
        tmp_path / "cavity", declarations_dir=_CASES / "postprocess_runtime", end_time=0.006
    )

    _, rows = read_table(case, "volume_p_run_time")

    assert [float(row[0]) for row in rows] == pytest.approx([0.002, 0.004, 0.006])


def test_a_case_declaring_no_table_writes_no_output_directory(tmp_path: Path) -> None:
    case = solved_cavity(tmp_path / "cavity")

    assert not (case.path / "postProcessing").exists()


def _core_model(spec_name: str, op_names: list[str]) -> SimpleNamespace:
    """A built core model as the graph step reads it: a spec name and named ops."""
    operations = [
        Operation(func=SequentialOp(lambda _ctx: None), metadata=OperationMetadata(op_name=name))
        for name in op_names
    ]
    return SimpleNamespace(
        spec=SimpleNamespace(name=spec_name),
        operations=operations,
        _build_operations_for=lambda _self: operations,
    )


def test_the_time_loop_ends_with_write_output_then_post_process() -> None:
    state = SimpleNamespace(
        core_models=[
            _core_model("pimple", ["inner_loop", "momentum", "continuity"]),
            _core_model("solutionLoop", ["set_time_step", "increment_time"]),
            _core_model("fieldWriter", ["write_output"]),
            _core_model("postProcess", ["post_process"]),
        ],
        optional_models=[],
    )
    ctx = Context(
        fields={},
        models={
            "viscosity": SimpleNamespace(operations=[]),
            "turbulence": SimpleNamespace(operations=[]),
        },
    )

    builder, _ = execution_graph(SimpleNamespace(state=state), ctx)

    time_loop = builder.operations["time_loop"]
    assert [op.operation_name for op in time_loop.sub_operations][-2:] == [
        "write_output",
        "post_process",
    ]
