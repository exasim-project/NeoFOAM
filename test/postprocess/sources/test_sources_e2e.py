# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: the ``patch`` and ``line`` sources against a solved lid-driven cavity.

**Integration, real OpenFOAM.** ``test_fields.py`` pins what a source *is*;
only a live mesh can pin what it *reads* — that a patch table sees the patch's
own boundary values on the patch's own face areas, and that a probe lands its
points where the spec put them. ``cases/sources/system/postProcess.yaml``
declares one table per source shape and the solver evaluates them all in one run.

**The mesh.** ``cases/cavity3x3/system/blockMeshDict`` (reused from
``test/solver/incompressibleFluid``) is 3x3x1 uniform cells in a
0.1 x 0.1 x 0.01 m box, with the lid on ``movingWall``. blockMesh numbers cells
with ``x`` fastest, so cells 6..8 are the row under the lid, and ``p`` is
``zeroGradient`` there — the patch value of a face *is* its cell value. That
makes every expectation a closed-form number off the written internal field,
computed without going through the code under test.

**The solved case.** Building the cavity, overlaying ``cases/sources`` and
solving it in a fresh interpreter is
:func:`solver.incompressibleFluid.solved_case.solved_cavity`, shared with the
solver-side ``test_post_process.py``; pytest puts ``test/`` on ``sys.path`` for
both packages, so it is imported by its package path rather than copied. The
reference read of ``p`` gets its own process again via ``CaseDir.read_field``.

**Tolerances.** ``_RTOL`` is the ASCII round-trip of the written ``0.003/p`` at
the helper's ``writePrecision 12``, with two decades of headroom. The probe's
end points lie *on* a wall, where ``cellPoint`` interpolation returns the
boundary condition itself and not a weighted sum, so those are compared with an
absolute tolerance that only covers the CSV round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from solver.incompressibleFluid.solved_case import read_table, solved_cavity

from neofoam.tooling.casebuild import CaseDir

_DECLARATION = Path(__file__).parents[1] / "cases" / "sources"

#: The blockMeshDict's box: 0.1 x 0.1 x 0.01 m.
_MESH_VOLUME = 1e-4

#: One lid face of the 3x3x1 mesh: (0.1 / 3) x 0.01 m.
_LID_FACE_AREA = 0.1 / 3 * 0.01

#: The whole lid: 0.1 x 0.01 m, and ``U`` on it is ``fixedValue (1 0 0)``.
_LID_INTEGRAL_U = (1e-3, 0.0, 0.0)

#: The cells under the lid, in blockMesh's ``x``-fastest numbering.
_LID_CELLS = slice(6, 9)

#: The last written time of the run below.
_LAST_TIME = "0.003"

#: See the module docstring: ASCII round-trip of ``writePrecision 12``.
_RTOL = 1e-10

#: The probe's end points sit on a wall, so their value is the BC verbatim.
_BC_ATOL = 1e-9


@pytest.fixture(scope="module")
def cavity_with_sources(tmp_path_factory: pytest.TempPathFactory) -> CaseDir:
    """The cavity solved for three steps with ``cases/sources`` declared on it."""
    dest = tmp_path_factory.mktemp("sources") / "cavity"
    return solved_cavity(dest, declarations_dir=_DECLARATION)


def _rows_at(case: CaseDir, name: str, time: str) -> list[list[str]]:
    """The table's data rows written at one time."""
    _, rows = read_table(case, name)
    return [row for row in rows if float(row[0]) == float(time)]


def test_summing_the_cell_measure_is_the_mesh_volume(cavity_with_sources: CaseDir) -> None:
    header, rows = read_table(cavity_with_sources, "mesh_volume")

    assert header == ["time", "mesh_volume"]
    assert float(rows[-1][1]) == pytest.approx(_MESH_VOLUME, rel=_RTOL)


def test_a_patch_table_writes_one_row_per_write_step(cavity_with_sources: CaseDir) -> None:
    header, rows = read_table(cavity_with_sources, "wall_p")

    assert header == ["time", "wall_p"]
    assert [float(row[0]) for row in rows] == pytest.approx([0.001, 0.002, 0.003])


def test_the_patch_row_is_p_on_the_lid_faces_times_their_area(
    cavity_with_sources: CaseDir,
) -> None:
    written_p: Any = cavity_with_sources.read_field("p", time=_LAST_TIME)

    _, rows = read_table(cavity_with_sources, "wall_p")

    assert float(rows[-1][1]) == pytest.approx(
        float(written_p[_LID_CELLS].sum()) * _LID_FACE_AREA, rel=_RTOL
    )


def test_the_patch_row_is_the_boundary_value_and_not_the_owner_cell_value(
    cavity_with_sources: CaseDir,
) -> None:
    # the lid's own value is (1 0 0) everywhere while the cells under it are
    # not, so only a boundary-value read integrates to the lid area exactly
    header, rows = read_table(cavity_with_sources, "wall_U")

    assert header == ["time", "wall_U_0", "wall_U_1", "wall_U_2"]
    assert [float(value) for value in rows[-1][1:]] == pytest.approx(
        _LID_INTEGRAL_U, rel=_RTOL, abs=_BC_ATOL
    )


def test_a_probe_writes_one_row_per_sample_point_per_write_step(
    cavity_with_sources: CaseDir,
) -> None:
    header, _ = read_table(cavity_with_sources, "u_profile")

    assert header == ["time", "x", "y", "z", "magU"]
    assert len(_rows_at(cavity_with_sources, "u_profile", _LAST_TIME)) == 5


def test_the_probe_end_points_carry_the_wall_boundary_conditions(
    cavity_with_sources: CaseDir,
) -> None:
    rows = _rows_at(cavity_with_sources, "u_profile", _LAST_TIME)

    bottom, top = rows[0], rows[-1]

    assert float(bottom[2]) == pytest.approx(0.0, abs=_BC_ATOL), "the no-slip wall"
    assert float(bottom[4]) == pytest.approx(0.0, abs=_BC_ATOL)
    assert float(top[2]) == pytest.approx(0.1, abs=_BC_ATOL), "the lid"
    assert float(top[4]) == pytest.approx(1.0, abs=_BC_ATOL)


def test_a_probe_that_leaves_the_mesh_writes_only_the_points_inside_it(
    cavity_with_sources: CaseDir,
) -> None:
    rows = _rows_at(cavity_with_sources, "u_profile_outside", _LAST_TIME)

    # the spec asks for five points spanning y = -0.05 .. 0.15
    assert 0 < len(rows) < 5
    assert all(0.0 <= float(row[2]) <= 0.1 for row in rows)
