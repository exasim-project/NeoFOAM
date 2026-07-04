# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Manifest -> mesh-dict builders (hermetic -- pydantic only, no OpenFOAM)."""

from neofoam.workflow.patch_set import PatchSet
from neofoam.workflow.mesh_inputs import block_mesh_dict, snappy_dict


def test_block_mesh_dict_box_spans_bbox(patch_set: PatchSet) -> None:
    """The single block's 8 vertices are the bbox corners."""
    cfg = block_mesh_dict(patch_set)
    lo, hi = patch_set.bbox.min, patch_set.bbox.max
    assert isinstance(cfg.vertices, list)
    assert set(cfg.vertices) == {
        (x, y, z)
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    }
    assert len(cfg.blocks) == 1


def test_block_mesh_dict_cells_from_length_scale(patch_set: PatchSet) -> None:
    """With no explicit sizing, cell counts come from half the length scale."""
    cfg = block_mesh_dict(patch_set)
    size = 0.5 * patch_set.length_scale
    lo, hi = patch_set.bbox.min, patch_set.bbox.max
    expected = tuple(max(1, round((hi[i] - lo[i]) / size)) for i in range(3))
    assert cfg.blocks[0].cells == expected


def test_block_mesh_dict_boundary_is_box_face_patches(patch_set: PatchSet) -> None:
    """Only box-face patches become blockMesh boundary patches (snappy carves the rest)."""
    cfg = block_mesh_dict(patch_set)
    assert isinstance(cfg.boundary, list)
    names = {p.name for p in cfg.boundary}
    box_face_patches = {p.name for p in patch_set.patches if p.box_faces}
    snappy_patches = {p.name for p in patch_set.patches if p.is_snappy_surface}
    assert names == box_face_patches
    assert names.isdisjoint(snappy_patches)


def test_block_mesh_dict_explicit_cells_override(patch_set: PatchSet) -> None:
    """An explicit ``cells`` argument wins over the length-scale heuristic."""
    cfg = block_mesh_dict(patch_set, cells=(10, 4, 1))
    assert cfg.blocks[0].cells == (10, 4, 1)


def test_snappy_dict_carves_only_snappy_surfaces(patch_set: PatchSet) -> None:
    """Each snappy-surface patch becomes a geometry + refinementSurfaces entry."""
    cfg = snappy_dict(patch_set)
    snappy_names = {p.name for p in patch_set.patches if p.is_snappy_surface}
    assert set(cfg.geometry) == snappy_names
    assert set(cfg.castellatedMeshControls["refinementSurfaces"]) == snappy_names
    for name in snappy_names:
        assert cfg.geometry[name]["type"] == "triSurfaceMesh"


def test_snappy_dict_location_in_mesh_from_manifest(patch_set: PatchSet) -> None:
    """``locationInMesh`` is the patch_set point, as an OpenFOAM literal."""
    cfg = snappy_dict(patch_set)
    x, y, z = patch_set.location_in_mesh
    assert cfg.castellatedMeshControls["locationInMesh"] == f"({x} {y} {z})"


def test_snappy_dict_uses_patch_surface_refinement(patch_set: PatchSet) -> None:
    """A patch's ``surface_refinement`` sets its refinement level."""
    cfg = snappy_dict(patch_set)
    for patch in patch_set.patches:
        if patch.is_snappy_surface and patch.surface_refinement:
            lo, hi = patch.surface_refinement
            level = cfg.castellatedMeshControls["refinementSurfaces"][patch.name][
                "level"
            ]
            assert level == f"({lo} {hi})"
