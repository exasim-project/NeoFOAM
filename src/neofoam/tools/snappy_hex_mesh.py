# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""snappyHexMesh tool — the build step *and* the ``system/snappyHexMeshDict`` writer.

Two things live here, both keyed to the ``snappyHexMesh`` tool:

* :class:`SnappyHexMeshStep` / :data:`snappyHexMeshTool` — the in-process
  ``@build`` initializer (refine/snap the prior mesh). It resolves the prior mesh
  from the live ``ctx`` at call time (never captured in a long-lived closure — that
  would create a mesh-bound reference cycle that segfaults across in-process runs).
  ``pyf`` / ``generate_snappy_hex_mesh`` are module globals so tests can
  monkeypatch them without building a mesh.

* :class:`SnappyHexMeshDictConfig` — a ``BaseConfig`` bound to
  ``system/snappyHexMeshDict`` that both **reads** an existing dict and **writes**
  one. ``snappyHexMeshDict`` is dict-structured (``geometry`` + control sub-dicts +
  scalars), which the OpenFOAM strategy round-trips faithfully — nested dicts
  recurse and compound leaves (``locationInMesh (x y z)`` / ``level (1 2)`` /
  ``features ( )``) are OpenFOAM-syntax strings that ``pybFoam`` re-tokenises — so
  no custom serializer is needed. A fresh dict is assembled via
  :meth:`SnappyHexMeshDictConfig.castellate_and_snap`.
"""

from typing import Any, List, Literal, Tuple

import pybFoam as pyf
from pybFoam.meshing import generate_snappy_hex_mesh
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neofoam.framework.initialization import InitStep, lazy
from neofoam.framework.tools import Tool
from neofoam.io import OF, BaseConfig, IOStrategy

from ._foam_tokens import point
from .registry import register_tool

# --------------------------------------------------------------------------- #
# The build step                                                              #
# --------------------------------------------------------------------------- #


class SnappyHexMeshStep(BaseModel):
    """Refine/snap the prior mesh in-process from ``snappyHexMeshDict``."""

    tool: Literal["snappyHexMesh"]
    dict_file: str = "system/snappyHexMeshDict"
    overwrite: bool = True
    verbose: bool = True


snappyHexMeshTool = Tool("snappyHexMesh", consumes_mesh=True)


@snappyHexMeshTool.build
def _build_snappy(cfg: SnappyHexMeshStep) -> list[InitStep]:
    def gen(ctx: dict[str, Any]) -> Any:
        mesh = ctx["_prev_mesh"]
        generate_snappy_hex_mesh(
            mesh,
            pyf.dictionary.read(cfg.dict_file),
            overwrite=cfg.overwrite,
            verbose=cfg.verbose,
        )
        return mesh

    return [lazy("preprocess.snappyHexMesh", gen)]


register_tool(snappyHexMeshTool)


# --------------------------------------------------------------------------- #
# The snappyHexMeshDict writer config                                         #
# --------------------------------------------------------------------------- #

# Fixed, sane control blocks (castellate + snap, no layers). These carry no
# geometry-dependent knobs, so they are emitted verbatim as native sub-dicts.
_SNAP_CONTROLS: dict[str, Any] = {
    "nSmoothPatch": 3,
    "tolerance": 2.0,
    "nSolveIter": 30,
    "nRelaxIter": 5,
    "nFeatureSnapIter": 10,
    "implicitFeatureSnap": True,
    "explicitFeatureSnap": False,
    "multiRegionFeatureSnap": False,
}

_ADD_LAYERS_CONTROLS: dict[str, Any] = {
    "relativeSizes": True,
    "layers": {},
    "expansionRatio": 1.2,
    "finalLayerThickness": 0.4,
    "minThickness": 0.1,
    "nGrow": 0,
    "featureAngle": 60,
    "nRelaxIter": 5,
    "nSmoothSurfaceNormals": 1,
    "nSmoothNormals": 3,
    "nSmoothThickness": 10,
    "maxFaceThicknessRatio": 0.5,
    "maxThicknessToMedialRatio": 0.3,
    "minMedialAxisAngle": 90,
    "nBufferCellsNoExtrude": 0,
    "nLayerIter": 50,
}

_MESH_QUALITY_CONTROLS: dict[str, Any] = {
    "maxNonOrtho": 65,
    "maxBoundarySkewness": 20,
    "maxInternalSkewness": 4,
    "maxConcave": 80,
    "minFlatness": 0.5,
    "minVol": 1e-13,
    "minTetQuality": 1e-15,
    "minArea": -1,
    "minTwist": 0.02,
    "minDeterminant": 0.001,
    "minFaceWeight": 0.05,
    "minVolRatio": 0.01,
    "minTriangleTwist": -1,
    "nSmoothScale": 4,
    "errorReduction": 0.75,
}


class SnappySurface(BaseModel):
    """One snappy ``geometry`` / ``refinementSurfaces`` surface backed by an STL."""

    name: str
    """snappy surface name = the resulting patch name (need not equal the file)."""
    file: str
    """STL basename under ``constant/triSurface`` (e.g. ``"tubes.stl"``)."""
    level: tuple[int, int] = (1, 2)
    """``(min, max)`` surface refinement levels."""
    patch_type: str = "wall"
    """``patchInfo`` type for the patch snappy creates (``wall`` / ``patch``)."""


@IOStrategy(OF("system/snappyHexMeshDict"))
class SnappyHexMeshDictConfig(BaseConfig):
    """``system/snappyHexMeshDict`` — a faithful load/write config.

    The top-level sections are declared as passthrough fields: ``geometry`` and
    the ``*Controls`` blocks are nested dictionaries the OpenFOAM strategy reads
    and writes verbatim (their compound leaves — ``locationInMesh``, ``level``,
    ``features`` — are OpenFOAM-syntax strings ``pybFoam`` re-tokenises), and
    unmodeled top-level keys are kept via ``extra="allow"``. So an existing dict
    round-trips (to ``pybFoam``'s 6-significant-figure write precision). Build a
    fresh castellate+snap dict with :meth:`castellate_and_snap` (used by
    :func:`neofoam.tooling.workflow.mesh_inputs.snappy_dict`).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    castellatedMesh: bool = True
    snap: bool = True
    addLayers: bool = False
    geometry: dict[str, Any] = Field(default_factory=dict)
    castellatedMeshControls: dict[str, Any] = Field(default_factory=dict)
    snapControls: dict[str, Any] = Field(default_factory=dict)
    addLayersControls: dict[str, Any] = Field(default_factory=dict)
    meshQualityControls: dict[str, Any] = Field(default_factory=dict)
    mergeTolerance: float = 1e-6

    @model_validator(mode="before")
    @classmethod
    def _drop_header(cls, data: Any) -> Any:
        """Drop the ``FoamFile`` header — the writer re-injects a fresh one."""
        if isinstance(data, dict) and "FoamFile" in data:
            return {k: v for k, v in data.items() if k != "FoamFile"}
        return data

    @classmethod
    def castellate_and_snap(
        cls,
        *,
        surfaces: List["SnappySurface"],
        location_in_mesh: Tuple[float, float, float],
        n_cells_between_levels: int = 2,
        resolve_feature_angle: float = 30.0,
        max_global_cells: int = 2_000_000,
        max_local_cells: int = 1_000_000,
        add_layers: bool = False,
    ) -> "SnappyHexMeshDictConfig":
        """Assemble a castellate+snap dict for a set of STL surfaces.

        Each surface becomes a ``geometry`` entry (``triSurfaceMesh``) and a
        ``refinementSurfaces`` entry; the snap / layer / quality control blocks are
        the fixed sane defaults.
        """
        geometry = {
            s.name: {"type": "triSurfaceMesh", "file": f'"{s.file}"'} for s in surfaces
        }
        refinement_surfaces = {
            s.name: {
                "level": f"({s.level[0]} {s.level[1]})",
                "patchInfo": {"type": s.patch_type},
            }
            for s in surfaces
        }
        return cls(
            castellatedMesh=True,
            snap=True,
            addLayers=add_layers,
            geometry=geometry,
            castellatedMeshControls={
                "maxLocalCells": max_local_cells,
                "maxGlobalCells": max_global_cells,
                "minRefinementCells": 0,
                "nCellsBetweenLevels": n_cells_between_levels,
                "maxLoadUnbalance": 0.10,
                "resolveFeatureAngle": resolve_feature_angle,
                "allowFreeStandingZoneFaces": True,
                "features": "( )",
                "refinementSurfaces": refinement_surfaces,
                "refinementRegions": {},
                "locationInMesh": point(location_in_mesh),
            },
            snapControls=dict(_SNAP_CONTROLS),
            addLayersControls=dict(_ADD_LAYERS_CONTROLS),
            meshQualityControls=dict(_MESH_QUALITY_CONTROLS),
            mergeTolerance=1e-6,
        )
