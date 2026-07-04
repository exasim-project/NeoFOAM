# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Derive the mesh-input dicts from the STL files in ``constant/triSurface`` (pure).

Self-contained geometry stage for the case wizard: given a case whose
``constant/triSurface/`` already holds the boundary STLs (they are a *given*
input — extraction is out of scope here), read each STL, classify it as either a
background-box face patch or a snappy refinement surface, and author
``system/blockMeshDict`` + ``system/snappyHexMeshDict`` + ``system/preprocess.yaml``.
The mesh itself is built later, in-process, when ``./Allrun`` runs the solver
against ``preprocess.yaml`` — this module only *writes the dicts*.

The dict-authoring math mirrors the deterministic ``PatchSet``→config mappers but
re-derives the geometry facts (bounding box, per-patch box faces) straight from the
STL triangles, so this module depends only on :mod:`neofoam.tools` (never on the
workflow package). No OpenFOAM, no LLM.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from neofoam.framework.tools import PreprocessConfig
from neofoam.io import write_configs
from neofoam.tools._foam_tokens import num
from neofoam.tools.block_mesh import Block, BlockMeshDictConfig, BlockPatch
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig, SnappySurface

__all__ = [
    "PatchRole",
    "BoxFace",
    "PatchGeometry",
    "GeometrySpec",
    "MeshSettings",
    "read_stl_bbox",
    "read_stl_vertices",
    "resolve_tri_dir",
    "discover_geometry",
    "block_mesh_dict",
    "snappy_dict",
    "preprocess_config",
    "write_mesh_configs",
]

Vec3 = tuple[float, float, float]


class PatchRole(str, Enum):
    """The CFD role of a boundary patch, used to pick its boundary condition."""

    inlet = "inlet"
    outlet = "outlet"
    wall = "wall"
    symmetry = "symmetry"
    empty = "empty"


BoxFace = Literal["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]

# The six background-box faces, as ``(axis, at_max)`` — axis 0/1/2 = x/y/z.
_FACES: tuple[tuple[BoxFace, int, bool], ...] = (
    ("x_min", 0, False),
    ("x_max", 0, True),
    ("y_min", 1, False),
    ("y_max", 1, True),
    ("z_min", 2, False),
    ("z_max", 2, True),
)

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

# A patch "owns" a box face when at least this fraction of its *triangles* are
# coplanar with the face (all three vertices on the face plane). Per-triangle (not
# per-vertex) is what distinguishes ownership from mere touching: a planar patch on
# x=0 has ~50% of its vertices on each perpendicular face (its edges) but *zero*
# triangles coplanar with them; a patch made of two opposite sheets (front+back)
# has ~50% of its triangles coplanar with each of the two faces.
_FACE_MEMBERSHIP = 0.1


@dataclass
class PatchGeometry:
    """One boundary patch backed by an STL under ``constant/triSurface``."""

    name: str
    """OpenFOAM patch name (the STL stem, e.g. ``"inlet"``)."""
    stl: str
    """STL basename, e.g. ``"inlet.stl"``."""
    role: PatchRole
    box_faces: Optional[list[BoxFace]] = None
    """Background-box faces this patch occupies, or ``None`` for a snappy surface."""
    refinement: Optional[tuple[int, int]] = None
    """Snappy ``(min, max)`` surface refinement (snappy surfaces only)."""

    @property
    def is_snappy_surface(self) -> bool:
        """True when the patch is meshed by snappy (owns no box faces)."""
        return not self.box_faces


@dataclass
class GeometrySpec:
    """Everything the mesh-input dicts need, derived from the triSurface STLs."""

    bbox_min: Vec3
    bbox_max: Vec3
    location_in_mesh: Vec3
    length_scale: float
    patches: list[PatchGeometry] = field(default_factory=list)

    def patch_names(self) -> list[str]:
        return [p.name for p in self.patches]


@dataclass
class MeshSettings:
    """Tunable background-mesh / snappy knobs for :func:`write_mesh_configs`."""

    cell_size: Optional[float] = None
    """Target background cell edge (m); defaults to half the ``length_scale``."""
    cells: Optional[tuple[int, int, int]] = None
    """Explicit ``(nx, ny, nz)`` override; wins over ``cell_size``."""
    padding: float = 0.0
    """Symmetric bbox expansion (m) per side (0 = box == bbox)."""
    grading: tuple[float, float, float] = (1.0, 1.0, 1.0)
    default_refinement: tuple[int, int] = (1, 2)
    n_cells_between_levels: int = 2
    resolve_feature_angle: float = 30.0
    max_global_cells: int = 2_000_000
    add_layers: bool = False


# --------------------------------------------------------------------------- #
# STL reading                                                                 #
# --------------------------------------------------------------------------- #


def read_stl_vertices(path: Path | str) -> list[Vec3]:
    """Every triangle vertex of an STL file (ASCII or binary), in file order.

    Binary is detected structurally (header + ``uint32`` triangle count exactly
    accounts for the file length), not by the leading ``solid`` token — a binary
    STL header may also start with ``solid``.
    """
    data = Path(path).read_bytes()
    if _is_binary_stl(data):
        return _read_binary_stl(data)
    return _read_ascii_stl(data.decode("utf-8", errors="replace"))


def read_stl_bbox(path: Path | str) -> tuple[Vec3, Vec3]:
    """Axis-aligned bounding box ``(min, max)`` of an STL file."""
    verts = read_stl_vertices(path)
    if not verts:
        raise ValueError(f"STL has no vertices: {path}")
    return _bbox(verts)


def _is_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    n_tri = int(struct.unpack_from("<I", data, 80)[0])
    return len(data) == 84 + n_tri * 50


def _read_binary_stl(data: bytes) -> list[Vec3]:
    n_tri = int(struct.unpack_from("<I", data, 80)[0])
    verts: list[Vec3] = []
    off = 84
    for _ in range(n_tri):
        # 12 floats: normal (3) + 3 vertices (9); trailing uint16 attribute count.
        vals = struct.unpack_from("<12f", data, off)
        verts.append((vals[3], vals[4], vals[5]))
        verts.append((vals[6], vals[7], vals[8]))
        verts.append((vals[9], vals[10], vals[11]))
        off += 50
    return verts


def _read_ascii_stl(text: str) -> list[Vec3]:
    verts: list[Vec3] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return verts


def _bbox(verts: list[Vec3]) -> tuple[Vec3, Vec3]:
    xs, ys, zs = zip(*verts)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# --------------------------------------------------------------------------- #
# Geometry discovery                                                          #
# --------------------------------------------------------------------------- #


def resolve_tri_dir(path: Path | str) -> Path:
    """Resolve ``path`` to the directory holding the STLs.

    Accepts either a triSurface directory directly (contains ``*.stl``) or a case
    directory (has ``constant/triSurface``), so the wizard can point straight at an
    STL folder.
    """
    p = Path(path)
    if p.is_dir() and any(p.glob("*.stl")):
        return p
    nested = p / "constant" / "triSurface"
    if nested.is_dir():
        return nested
    raise ValueError(f"no STL files at {p} (nor {nested})")


def discover_geometry(
    case_dir: Path | str,
    *,
    default_refinement: tuple[int, int] = (1, 2),
) -> GeometrySpec:
    """Read the STLs at ``case_dir`` (an STL folder or a case dir) into patches.

    The union of all STL bounding boxes is the background domain. A patch that
    lies on one or more domain faces (a planar, axis-aligned surface) becomes a
    background-box patch owning those faces; any other patch (interior geometry,
    e.g. a tube bank) becomes a snappy refinement surface. Patch roles are a
    filename/face heuristic — editable downstream in the wizard.
    """
    tri_dir = resolve_tri_dir(case_dir)
    stl_paths = sorted(tri_dir.glob("*.stl"))
    if not stl_paths:
        raise ValueError(f"no STL files under {tri_dir}")

    per_patch = [(p.stem, p.name, read_stl_vertices(p)) for p in stl_paths]
    all_verts = [v for _, _, verts in per_patch for v in verts]
    dmin, dmax = _bbox(all_verts)

    extents = [dmax[i] - dmin[i] for i in range(3)]
    positive = [e for e in extents if e > 0.0]
    length_scale = min(positive) if positive else max(extents) or 1.0
    tol = 1e-5 * (max(extents) or 1.0)

    patches: list[PatchGeometry] = []
    for name, stl, verts in per_patch:
        faces = _owned_faces(_triangles(verts), dmin, dmax, tol)
        role = _role_for(name, faces)
        patches.append(
            PatchGeometry(
                name=name,
                stl=stl,
                role=role,
                box_faces=faces or None,
                refinement=None if faces else default_refinement,
            )
        )

    location = tuple(0.5 * (dmin[i] + dmax[i]) for i in range(3))
    return GeometrySpec(
        bbox_min=dmin,
        bbox_max=dmax,
        location_in_mesh=(location[0], location[1], location[2]),
        length_scale=length_scale,
        patches=patches,
    )


def _triangles(verts: list[Vec3]) -> list[tuple[Vec3, Vec3, Vec3]]:
    """Group a flat vertex list into triangles (3 vertices each, in file order)."""
    return [(verts[i], verts[i + 1], verts[i + 2]) for i in range(0, len(verts) - 2, 3)]


def _owned_faces(
    triangles: list[tuple[Vec3, Vec3, Vec3]], dmin: Vec3, dmax: Vec3, tol: float
) -> list[BoxFace]:
    """Domain faces a patch lies on (fraction of its triangles coplanar with each)."""
    if not triangles:
        return []
    owned: list[BoxFace] = []
    n = len(triangles)
    for face, axis, at_max in _FACES:
        plane = dmax[axis] if at_max else dmin[axis]
        coplanar = sum(
            1 for tri in triangles if all(abs(v[axis] - plane) <= tol for v in tri)
        )
        if coplanar / n >= _FACE_MEMBERSHIP:
            owned.append(face)
    return owned


def _role_for(name: str, faces: list[BoxFace]) -> PatchRole:
    """Heuristic role from the STL name, then the owned box faces."""
    low = name.lower()
    if "inlet" in low:
        return PatchRole.inlet
    if "outlet" in low:
        return PatchRole.outlet
    if "wall" in low:
        return PatchRole.wall
    face_set = set(faces)
    if face_set == {"x_min"}:
        return PatchRole.inlet
    if face_set == {"x_max"}:
        return PatchRole.outlet
    if face_set and face_set <= {"z_min", "z_max"}:
        return PatchRole.empty
    if face_set:
        return PatchRole.wall
    return PatchRole.wall  # interior snappy surface


# --------------------------------------------------------------------------- #
# Config authoring                                                            #
# --------------------------------------------------------------------------- #


def _resolve_cells(spec: GeometrySpec, settings: MeshSettings) -> tuple[int, int, int]:
    if settings.cells is not None:
        return settings.cells
    size = settings.cell_size or 0.5 * spec.length_scale
    lo, hi, pad = spec.bbox_min, spec.bbox_max, settings.padding
    return (
        max(1, round((hi[0] - lo[0] + 2 * pad) / size)),
        max(1, round((hi[1] - lo[1] + 2 * pad) / size)),
        max(1, round((hi[2] - lo[2] + 2 * pad) / size)),
    )


def block_mesh_dict(spec: GeometrySpec, settings: MeshSettings) -> BlockMeshDictConfig:
    """One graded hex enclosing the bbox; box-face patches named from ``spec``."""
    pad = settings.padding
    lo = [spec.bbox_min[i] - pad for i in range(3)]
    hi = [spec.bbox_max[i] + pad for i in range(3)]
    vertices: list[Vec3] = [
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
        for patch in spec.patches
        if patch.box_faces
    ]
    gx, gy, gz = settings.grading
    block = Block(
        vertices=list(range(8)),
        cells=_resolve_cells(spec, settings),
        grading=f"simpleGrading ( {num(gx)} {num(gy)} {num(gz)} )",
    )
    return BlockMeshDictConfig(vertices=vertices, blocks=[block], boundary=boundary)


def snappy_dict(spec: GeometrySpec, settings: MeshSettings) -> SnappyHexMeshDictConfig:
    """Carve the snappy-surface patches; ``locationInMesh`` from ``spec``."""
    surfaces = [
        SnappySurface(
            name=patch.name,
            file=patch.stl,
            level=patch.refinement or settings.default_refinement,
            patch_type=_PATCH_TYPE[patch.role],
        )
        for patch in spec.patches
        if patch.is_snappy_surface
    ]
    return SnappyHexMeshDictConfig.castellate_and_snap(
        surfaces=surfaces,
        location_in_mesh=spec.location_in_mesh,
        n_cells_between_levels=settings.n_cells_between_levels,
        resolve_feature_angle=settings.resolve_feature_angle,
        max_global_cells=settings.max_global_cells,
        add_layers=settings.add_layers,
    )


def preprocess_config(*, with_snappy: bool = True) -> PreprocessConfig:
    """The ``system/preprocess.yaml`` tool DAG: blockMesh → snappyHexMesh → checkMesh."""
    tools: list[dict[str, object]] = [{"tool": "blockMesh"}]
    if with_snappy:
        tools.append({"tool": "snappyHexMesh", "depends_on": ["blockMesh"]})
        tools.append({"tool": "checkMesh", "depends_on": ["snappyHexMesh"]})
    else:
        tools.append({"tool": "checkMesh", "depends_on": ["blockMesh"]})
    return PreprocessConfig(tools=tools)


def write_mesh_configs(
    case_dir: Path | str,
    spec: GeometrySpec,
    settings: Optional[MeshSettings] = None,
) -> list[Path]:
    """Author ``blockMeshDict`` + ``snappyHexMeshDict`` + ``preprocess.yaml``.

    Returns the paths written. The snappy dict / snappy DAG step are emitted only
    when the geometry actually has a snappy surface, so a pure box-mesh case skips
    ``snappyHexMesh``.
    """
    settings = settings or MeshSettings()
    case = Path(case_dir)
    has_snappy = any(p.is_snappy_surface for p in spec.patches)

    of_configs: list[BlockMeshDictConfig | SnappyHexMeshDictConfig] = [
        block_mesh_dict(spec, settings)
    ]
    if has_snappy:
        of_configs.append(snappy_dict(spec, settings))
    # OpenFOAM dict configs (FoamFile header injected) via the merged writer.
    written_map = write_configs(of_configs, case)

    # preprocess.yaml is a YAMLStrategy config — outside write_configs' merged path.
    preprocess_config(with_snappy=has_snappy).save(case_dir=case)

    written = [case / rel for rel in written_map]
    written.append(case / "system" / "preprocess.yaml")
    return written
