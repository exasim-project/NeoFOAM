# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Build the two mesh-input configs from a :class:`PatchSet`.

Deterministic patch_set → ``BaseConfig`` mappers (no OpenFOAM, no LLM):

* :func:`block_mesh_dict` — one hex enclosing the bbox; each patch_set patch with
  ``box_faces`` becomes a blockMesh ``boundary`` patch owning those box faces.
* :func:`snappy_dict` — each snappy-surface patch (``box_faces is None``) becomes
  a ``geometry`` + ``refinementSurfaces`` entry; ``locationInMesh`` comes from the
  patch_set.
* :func:`build_mesh_inputs` — the one-call bridge: returns the ``blockMeshDict``,
  a ``snappyHexMeshDict`` **only when** some patch is a snappy surface, and the
  matching ``preprocess.yaml`` enable-list (blockMesh → [snappyHexMesh] →
  checkMesh). This is what the MCP ``build_mesh_inputs`` tool writes so a case that
  staged a manifest can be meshed without hand-authoring the mesh dicts.

The resulting configs serialise to ``system/blockMeshDict`` /
``system/snappyHexMeshDict`` / ``system/preprocess.yaml`` via
:func:`neofoam.io.write_configs` — the patch_set supplies the geometry facts, the
configs know how to write themselves.
"""

from __future__ import annotations

from pathlib import Path

from neofoam.framework.tools.graph import PreprocessConfig
from neofoam.tooling.workflow.patch_set import BoxFace, PatchSet, PatchRole
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
    patch_set: PatchSet,
    cell_size: float | None,
    cells: tuple[int, int, int] | None,
    padding: float,
) -> tuple[int, int, int]:
    """Background cell counts, from explicit ``cells`` or a target ``cell_size``."""
    if cells is not None:
        return cells
    size = cell_size or 0.5 * patch_set.length_scale
    lo, hi = patch_set.bbox.min, patch_set.bbox.max
    return (
        max(1, round((hi[0] - lo[0] + 2 * padding) / size)),
        max(1, round((hi[1] - lo[1] + 2 * padding) / size)),
        max(1, round((hi[2] - lo[2] + 2 * padding) / size)),
    )


def block_mesh_dict(
    patch_set: PatchSet,
    *,
    cell_size: float | None = None,
    cells: tuple[int, int, int] | None = None,
    padding: float = 0.0,
    grading: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> BlockMeshDictConfig:
    """One graded hex box enclosing the bbox, with the box-face patches named.

    Args:
        patch_set: geometry facts (bbox + patches).
        cell_size: target background cell edge length (m); defaults to half the
            patch_set ``length_scale`` so the smallest feature spans ~2 base cells.
        cells: explicit ``(nx, ny, nz)`` override; takes precedence over
            ``cell_size``.
        padding: symmetric bbox expansion (m) per side (0 = box == bbox).
        grading: ``simpleGrading`` ratios.
    """
    lo = [patch_set.bbox.min[i] - padding for i in range(3)]
    hi = [patch_set.bbox.max[i] + padding for i in range(3)]
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
        for patch in patch_set.patches
        if patch.box_faces
    ]
    gx, gy, gz = grading
    block = Block(
        vertices=list(range(8)),
        cells=_resolve_cells(patch_set, cell_size, cells, padding),
        grading=f"simpleGrading ( {num(gx)} {num(gy)} {num(gz)} )",
    )
    return BlockMeshDictConfig(vertices=vertices, blocks=[block], boundary=boundary)


def snappy_dict(
    patch_set: PatchSet,
    *,
    default_level: tuple[int, int] = (1, 2),
    n_cells_between_levels: int = 2,
    resolve_feature_angle: float = 30.0,
    max_global_cells: int = 2_000_000,
    add_layers: bool = False,
) -> SnappyHexMeshDictConfig:
    """Carve the snappy-surface patches; ``locationInMesh`` from the patch_set.

    Each patch_set patch with no ``box_faces`` (a snappy surface) becomes a
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
        for patch in patch_set.patches
        if patch.is_snappy_surface
    ]
    return SnappyHexMeshDictConfig.castellate_and_snap(
        surfaces=surfaces,
        location_in_mesh=patch_set.location_in_mesh,
        n_cells_between_levels=n_cells_between_levels,
        resolve_feature_angle=resolve_feature_angle,
        max_global_cells=max_global_cells,
        add_layers=add_layers,
    )


def _has_snappy_surface(patch_set: PatchSet) -> bool:
    return any(p.is_snappy_surface for p in patch_set.patches)


def preprocess_config(*, has_snappy: bool) -> PreprocessConfig:
    """The ``preprocess.yaml`` enable-list: blockMesh → [snappyHexMesh] → checkMesh.

    ``checkMesh`` runs last and is non-fatal (``fail_on_error: false``) so a mesh
    that only warns still lets the solver run. ``snappyHexMesh`` is inserted (after
    ``blockMesh``) only when the manifest has a snappy surface.
    """
    tools: list[dict[str, object]] = [{"tool": "blockMesh"}]
    check_deps = ["blockMesh"]
    if has_snappy:
        tools.append({"tool": "snappyHexMesh", "depends_on": ["blockMesh"]})
        check_deps = ["snappyHexMesh"]
    tools.append(
        {"tool": "checkMesh", "depends_on": check_deps, "fail_on_error": False}
    )
    return PreprocessConfig(tools=tools)


def build_mesh_inputs(
    patch_set: PatchSet,
    *,
    cell_size: float | None = None,
    cells: tuple[int, int, int] | None = None,
    padding: float = 0.0,
) -> tuple[BlockMeshDictConfig, SnappyHexMeshDictConfig | None, PreprocessConfig]:
    """Render ``(blockMeshDict, snappyHexMeshDict | None, preprocess)`` from a manifest.

    The one-call manifest → mesh-dict bridge behind the MCP ``build_mesh_inputs``
    tool. The snappy dict is ``None`` for a pure-box case (every patch is a
    blockMesh face), and the ``preprocess`` enable-list is sized to match (it
    includes ``snappyHexMesh`` only when a snappy dict is produced).
    """
    block = block_mesh_dict(
        patch_set, cell_size=cell_size, cells=cells, padding=padding
    )
    has_snappy = _has_snappy_surface(patch_set)
    snappy = snappy_dict(patch_set) if has_snappy else None
    pre = preprocess_config(has_snappy=has_snappy)
    return block, snappy, pre
