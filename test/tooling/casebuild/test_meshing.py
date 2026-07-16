# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the injectable meshing strategies: ``block_mesh``, ``box``, ``snappy_hex_mesh``.

Each runs a real in-process mesh via the reused ``neofoam.tools`` mesh tools (pybFoam is
a hard dep — no skip). Point counts are exact for structured block meshes
(``(nx+1)(ny+1)(nz+1)``), so those are the assertions; snappy is proven by refinement
(more points than the background mesh). Committed cases are staged into ``tmp_path``.
"""

from pathlib import Path

from neofoam.tooling.casebuild import block_mesh, box, from_template, snappy_hex_mesh

CASES = Path(__file__).parent / "cases"
CAVITY = CASES / "cavity"
SNAPPY = CASES / "snappy"


def _n_points(case_path: Path) -> int:
    """Read the point count from ``constant/polyMesh/points`` (first bare-integer line)."""
    for line in (
        (case_path / "constant" / "polyMesh" / "points").read_text().splitlines()
    ):
        stripped = line.strip()
        if stripped.isdigit():
            return int(stripped)
    raise AssertionError("no point count in points file")


def test_block_mesh_generates_polymesh_from_committed_dict(tmp_path: Path) -> None:
    case = (from_template(CAVITY) | block_mesh()).build_at(tmp_path / "c")
    assert (case.path / "constant" / "polyMesh" / "points").is_file()
    assert _n_points(case.path) == (3 + 1) * (3 + 1) * (1 + 1)  # committed 3x3x1 dict


def test_box_synthesizes_dict_at_requested_resolution(tmp_path: Path) -> None:
    # box overwrites the committed 3x3x1 blockMeshDict with a 5x5x1 one.
    case = (from_template(CAVITY) | box(n=(5, 5, 1))).build_at(tmp_path / "c")
    assert _n_points(case.path) == (5 + 1) * (5 + 1) * (
        1 + 1
    )  # 72, not the committed 32


def test_snappy_refines_the_background_mesh(tmp_path: Path) -> None:
    background = (from_template(SNAPPY) | block_mesh()).build_at(tmp_path / "bg")
    refined = (from_template(SNAPPY) | block_mesh() | snappy_hex_mesh()).build_at(
        tmp_path / "snap"
    )
    assert _n_points(refined.path) > _n_points(background.path)
