# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The patch manifest -- the contract handed across every e2e stage.

Stage 1 (STL extraction) runs in a *separate* conda env (FreeCAD) and writes
``<case>/manifest.json``; stages 2 and 3 (mesh authoring, boundary-condition
filling) run in the neofoam env and read it back. The filesystem is therefore
the cross-env boundary, and this module is the single schema both sides agree
on. All coordinates are in **metres** (the extractor applies ``scale_to_meters``
when exporting the STL, so the manifest and the triSurface STLs share one
coordinate system).

The module imports only pydantic + stdlib so it stays importable from a minimal
environment and the stage-1 unit test is hermetic.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]


class PatchRole(str, Enum):
    """The CFD role of a boundary patch, used to pick its boundary condition."""

    inlet = "inlet"
    outlet = "outlet"
    wall = "wall"
    symmetry = "symmetry"
    empty = "empty"


BoxFace = Literal["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]


class PatchEntry(BaseModel):
    """One named boundary patch backed by a triangulated surface."""

    name: str
    """OpenFOAM patch name, e.g. ``"inlet"``."""
    stl: str
    """STL path relative to ``case_dir``, e.g. ``constant/triSurface/inlet.stl``."""
    role: PatchRole
    box_faces: list[BoxFace] | None = None
    """Which background-box faces this patch occupies (planar, axis-aligned
    patches), or ``None`` for a snappy surface meshed from its STL. Stage 2
    realises ``box_faces`` patches as blockMesh boundary patches and ``None``
    patches as snappy ``refinementSurfaces``."""
    surface_refinement: tuple[int, int] | None = None
    """Optional snappy ``(min, max)`` surface refinement levels for this patch."""

    @property
    def is_snappy_surface(self) -> bool:
        """True if this patch is meshed by snappy (no background-box faces)."""
        return not self.box_faces


class BoundingBox(BaseModel):
    """Axis-aligned bounding box of the geometry, in metres."""

    min: Vec3
    max: Vec3


class PatchManifest(BaseModel):
    """Everything stages 2 and 3 need to know about the extracted geometry."""

    case_dir: str
    geometry_source: str
    source_units: str = "mm"
    """Native CAD units; recorded for traceability."""
    scale_to_meters: float = 1.0
    """Factor applied to native coordinates when exporting (mm -> 0.001)."""
    bbox: BoundingBox
    location_in_mesh: Vec3
    """A point inside the fluid region (metres) -- snappy's ``locationInMesh``."""
    length_scale: float
    """Characteristic length (metres) for background-mesh / refinement sizing."""
    patches: list[PatchEntry] = Field(default_factory=list)

    # -- io -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> PatchManifest:
        """Load and validate a manifest from ``manifest.json``."""
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> Path:
        """Write the manifest as pretty JSON; return the path written."""
        out = Path(path)
        out.write_text(self.model_dump_json(indent=2))
        return out

    # -- queries ------------------------------------------------------------
    def patch_names(self) -> list[str]:
        """Patch names in declared order."""
        return [p.name for p in self.patches]

    def by_role(self, role: PatchRole) -> list[PatchEntry]:
        """All patches with the given role."""
        return [p for p in self.patches if p.role == role]
