# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Geometry staging: copy the extracted STLs into a case (hermetic -- pure IO)."""

from pathlib import Path

import pytest

from neofoam.e2e.geometry import stage_geometry
from neofoam.e2e.manifest import PatchManifest

TRISURFACE = Path(__file__).parent / "constant" / "triSurface"


def test_stage_geometry_lands_manifest_surfaces(
    manifest: PatchManifest, tmp_path: Path
) -> None:
    """Every surface the manifest declares lands where its ``stl`` path points."""
    stage_geometry(TRISURFACE, tmp_path, manifest=manifest)
    for patch in manifest.patches:
        target = tmp_path / patch.stl
        assert target.is_file(), f"missing staged surface: {patch.stl}"
        assert target.read_bytes() == (TRISURFACE / target.name).read_bytes()


def test_stage_geometry_returns_copied_paths(
    manifest: PatchManifest, tmp_path: Path
) -> None:
    """The return value is the destination paths, one per manifest surface."""
    copied = stage_geometry(TRISURFACE, tmp_path, manifest=manifest)
    assert [p.name for p in copied] == [Path(p.stl).name for p in manifest.patches]


def test_stage_geometry_is_idempotent(manifest: PatchManifest, tmp_path: Path) -> None:
    """Re-staging overwrites in place -- no duplicates, no error."""
    stage_geometry(TRISURFACE, tmp_path, manifest=manifest)
    stage_geometry(TRISURFACE, tmp_path, manifest=manifest)
    staged = sorted((tmp_path / "constant" / "triSurface").glob("*.stl"))
    assert len(staged) == len(manifest.patches)


def test_stage_geometry_without_manifest_copies_all_stls(tmp_path: Path) -> None:
    """Without a manifest, every ``*.stl`` in the source directory is copied."""
    copied = stage_geometry(TRISURFACE, tmp_path)
    assert {p.name for p in copied} == {p.name for p in TRISURFACE.glob("*.stl")}


def test_stage_geometry_missing_surface_raises(
    manifest: PatchManifest, tmp_path: Path
) -> None:
    """A manifest surface absent from the source fails fast."""
    empty = tmp_path / "empty_src"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        stage_geometry(empty, tmp_path / "case", manifest=manifest)
