# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Headless tests for the STL → mesh-dict geometry stage.

Uses the *real* tube-bank triSurface STLs (a given input) and round-trips the
authored dicts through their reader configs. Needs pybFoam because writing/reading
the OpenFOAM mesh dicts goes through the native dictionary I/O.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.framework.tools import PreprocessConfig  # noqa: E402
from neofoam.tools.block_mesh import BlockMeshDictConfig  # noqa: E402
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig  # noqa: E402
from neofoam.ui.geometry import (  # noqa: E402
    MeshSettings,
    PatchRole,
    discover_geometry,
    read_stl_bbox,
    read_stl_vertices,
    resolve_tri_dir,
    write_mesh_configs,
)

_TUBE_BANK = Path(__file__).resolve().parents[1] / "e2e" / "cases" / "tube_bank"
_TRI = _TUBE_BANK / "constant" / "triSurface"


def test_read_stl_bbox_ascii():
    lo, hi = read_stl_bbox(_TRI / "inlet.stl")
    assert lo == pytest.approx((0.0, 0.0, 0.0))
    assert hi == pytest.approx((0.0, 0.16, 0.02))


def _write_binary_stl(path: Path, tris: list[tuple]) -> None:
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            fh.write(struct.pack("<12fH", 0, 0, 0, *a, *b, *c, 0))


def test_read_stl_bbox_binary(tmp_path):
    p = tmp_path / "tri.stl"
    _write_binary_stl(p, [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0))])
    assert len(read_stl_vertices(p)) == 3
    lo, hi = read_stl_bbox(p)
    assert lo == pytest.approx((0.0, 0.0, 0.0))
    assert hi == pytest.approx((1.0, 2.0, 0.0))


def test_resolve_tri_dir_accepts_case_or_stl_folder():
    # Case dir → its constant/triSurface.
    assert resolve_tri_dir(_TUBE_BANK) == _TRI
    # A triSurface folder directly → itself.
    assert resolve_tri_dir(_TRI) == _TRI


def test_discover_geometry_from_stl_folder_directly():
    # Passing the triSurface folder itself works (the UI's STL-folder field).
    spec = discover_geometry(_TRI)
    assert {p.name for p in spec.patches} == {
        "inlet",
        "outlet",
        "walls",
        "frontBack",
        "tubes",
    }


def test_discover_geometry_classifies_tube_bank():
    spec = discover_geometry(_TUBE_BANK)

    assert spec.bbox_min == pytest.approx((0.0, 0.0, 0.0))
    assert spec.bbox_max == pytest.approx((0.576, 0.16, 0.02))

    by = {p.name: p for p in spec.patches}
    assert set(by) == {"inlet", "outlet", "walls", "frontBack", "tubes"}

    assert by["inlet"].box_faces == ["x_min"]
    assert by["inlet"].role is PatchRole.inlet
    assert by["outlet"].box_faces == ["x_max"]
    assert by["outlet"].role is PatchRole.outlet
    assert set(by["walls"].box_faces) == {"y_min", "y_max"}
    assert by["walls"].role is PatchRole.wall
    assert set(by["frontBack"].box_faces) == {"z_min", "z_max"}
    assert by["frontBack"].role is PatchRole.empty

    # tubes is interior geometry → a snappy refinement surface, not a box face.
    assert by["tubes"].is_snappy_surface
    assert by["tubes"].box_faces is None
    assert by["tubes"].role is PatchRole.wall
    assert by["tubes"].refinement == (1, 2)

    # locationInMesh is inside the domain; length_scale is the thin (z) extent.
    assert spec.length_scale == pytest.approx(0.02)


def test_write_mesh_configs_roundtrip(tmp_path):
    spec = discover_geometry(_TUBE_BANK)
    written = write_mesh_configs(tmp_path, spec, MeshSettings(cells=(120, 33, 4)))

    assert {Path(p).name for p in written} == {
        "blockMeshDict",
        "snappyHexMeshDict",
        "preprocess.yaml",
    }

    bmd = BlockMeshDictConfig.load(case_dir=tmp_path)
    assert isinstance(bmd.boundary, list)
    types = {p.name: p.type for p in bmd.boundary}
    # Only the box-face patches land in blockMeshDict (tubes is meshed by snappy).
    assert set(types) == {"inlet", "outlet", "walls", "frontBack"}
    assert types["inlet"] == "patch"
    assert types["outlet"] == "patch"
    assert types["walls"] == "wall"
    assert types["frontBack"] == "symmetry"

    shm = SnappyHexMeshDictConfig.load(case_dir=tmp_path)
    assert "tubes" in shm.geometry
    surfaces = shm.castellatedMeshControls["refinementSurfaces"]
    assert "tubes" in surfaces

    pre = PreprocessConfig.load(case_dir=tmp_path)
    assert [t["tool"] for t in pre.tools] == ["blockMesh", "snappyHexMesh", "checkMesh"]


def test_write_mesh_configs_skips_snappy_without_surface(tmp_path):
    # A pure box geometry (no interior surface) → no snappyHexMeshDict, no snappy step.
    spec = discover_geometry(_TUBE_BANK)
    spec.patches = [p for p in spec.patches if not p.is_snappy_surface]
    written = write_mesh_configs(tmp_path, spec)

    assert {Path(p).name for p in written} == {"blockMeshDict", "preprocess.yaml"}
    pre = PreprocessConfig.load(case_dir=tmp_path)
    assert [t["tool"] for t in pre.tools] == ["blockMesh", "checkMesh"]
