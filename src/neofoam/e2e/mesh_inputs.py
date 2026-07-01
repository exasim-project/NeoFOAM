# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Build the two mesh-input configs from a :class:`PatchManifest`.

Deterministic manifest → ``BaseConfig`` mappers (no OpenFOAM, no LLM):

* :func:`block_mesh_dict` — one hex enclosing the bbox; each manifest patch with
  ``box_faces`` becomes a blockMesh ``boundary`` patch owning those box faces.
* :func:`snappy_dict` — each snappy-surface patch (``box_faces is None``) becomes
  a ``geometry`` + ``refinementSurfaces`` entry; ``locationInMesh`` comes from the
  manifest.

The resulting configs serialise to ``system/blockMeshDict`` /
``system/snappyHexMeshDict`` via :func:`neofoam.io.write_configs` — the manifest
supplies the geometry facts, the configs know how to write themselves.
"""

from __future__ import annotations

from pathlib import Path

from neofoam.e2e.manifest import BoxFace, PatchManifest, PatchRole
from neofoam.tools._foam_tokens import num
from neofoam.tools.block_mesh import Block, BlockMeshDictConfig, BlockPatch
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig, SnappySurface

# Hex vertex layout (blockMesh convention) and the outward quad of each box face.
#   0:(xmin,ymin,zmin) 1:(xmax,ymin,zmin) 2:(xmax,ymax,zmin) 3:(xmin,ymax,zmin)
#   4:(xmin,ymin,zmax) 5:(xmax,ymin,zmax) 6:(xmax,ymax,zmax) 7:(xmin,ymax,zmax)
_FACE_QUADS: dict[BoxFace, tuple[int, int, int, int]] = {
    "x_min": (0, 4, 7, 3),
    "x_max": (1, 2, 6, 5),
    "y_min": (0, 1, 5, 4),
    "y_max": (3, 7, 6, 2),
    "z_min": (0, 3, 2, 1),
    "z_max": (4, 5, 6, 7),
}

# A snappy 3-D mesh cannot honour a true ``empty`` (2-D) patch, so the thin-slab
# redundant direction is realised as a generalized ``symmetry`` patch (not
# ``symmetryPlane``: its two opposite faces live in one patch).
_PATCH_TYPE: dict[PatchRole, str] = {
    PatchRole.inlet: "patch",
    PatchRole.outlet: "patch",
    PatchRole.wall: "wall",
    PatchRole.symmetry: "symmetry",
    PatchRole.empty: "symmetry",
}


def _resolve_cells(
    manifest: PatchManifest,
    cell_size: float | None,
    cells: tuple[int, int, int] | None,
    padding: float,
) -> tuple[int, int, int]:
    """Background cell counts, from explicit ``cells`` or a target ``cell_size``."""
    if cells is not None:
        return cells
    size = cell_size or 0.5 * manifest.length_scale
    lo, hi = manifest.bbox.min, manifest.bbox.max
    return (
        max(1, round((hi[0] - lo[0] + 2 * padding) / size)),
        max(1, round((hi[1] - lo[1] + 2 * padding) / size)),
        max(1, round((hi[2] - lo[2] + 2 * padding) / size)),
    )


def block_mesh_dict(
    manifest: PatchManifest,
    *,
    cell_size: float | None = None,
    cells: tuple[int, int, int] | None = None,
    padding: float = 0.0,
    grading: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> BlockMeshDictConfig:
    """One graded hex box enclosing the bbox, with the box-face patches named.

    Args:
        manifest: geometry facts (bbox + patches).
        cell_size: target background cell edge length (m); defaults to half the
            manifest ``length_scale`` so the smallest feature spans ~2 base cells.
        cells: explicit ``(nx, ny, nz)`` override; takes precedence over
            ``cell_size``.
        padding: symmetric bbox expansion (m) per side (0 = box == bbox).
        grading: ``simpleGrading`` ratios.
    """
    lo = [manifest.bbox.min[i] - padding for i in range(3)]
    hi = [manifest.bbox.max[i] + padding for i in range(3)]
    vertices: list[tuple[float, float, float]] = [
        (lo[0], lo[1], lo[2]),
        (hi[0], lo[1], lo[2]),
        (hi[0], hi[1], lo[2]),
        (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]),
        (hi[0], lo[1], hi[2]),
        (hi[0], hi[1], hi[2]),
        (lo[0], hi[1], hi[2]),
    ]
    boundary = [
        BlockPatch(
            name=patch.name,
            type=_PATCH_TYPE[patch.role],
            faces=[_FACE_QUADS[f] for f in patch.box_faces],
        )
        for patch in manifest.patches
        if patch.box_faces
    ]
    gx, gy, gz = grading
    block = Block(
        vertices=list(range(8)),
        cells=_resolve_cells(manifest, cell_size, cells, padding),
        grading=f"simpleGrading ( {num(gx)} {num(gy)} {num(gz)} )",
    )
    return BlockMeshDictConfig(vertices=vertices, blocks=[block], boundary=boundary)


def snappy_dict(
    manifest: PatchManifest,
    *,
    default_level: tuple[int, int] = (1, 2),
    n_cells_between_levels: int = 2,
    resolve_feature_angle: float = 30.0,
    max_global_cells: int = 2_000_000,
    add_layers: bool = False,
) -> SnappyHexMeshDictConfig:
    """Carve the snappy-surface patches; ``locationInMesh`` from the manifest.

    Each manifest patch with no ``box_faces`` (a snappy surface) becomes a
    ``geometry`` + ``refinementSurfaces`` entry backed by its STL basename; its
    ``surface_refinement`` overrides ``default_level`` when set.
    """
    surfaces = [
        SnappySurface(
            name=patch.name,
            file=Path(patch.stl).name,
            level=patch.surface_refinement or default_level,
            patch_type=_PATCH_TYPE[patch.role],
        )
        for patch in manifest.patches
        if patch.is_snappy_surface
    ]
    return SnappyHexMeshDictConfig.castellate_and_snap(
        surfaces=surfaces,
        location_in_mesh=manifest.location_in_mesh,
        n_cells_between_levels=n_cells_between_levels,
        resolve_feature_angle=resolve_feature_angle,
        max_global_cells=max_global_cells,
        add_layers=add_layers,
    )
