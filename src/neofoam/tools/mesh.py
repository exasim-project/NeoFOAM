# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared in-process mesh tools: blockMesh, snappyHexMesh, checkMesh.

Each is a :class:`Tool`; its ``@build`` emits a single :class:`InitStep`. The
initializers resolve pybFoam objects from the live ``ctx`` at call time (never captured
in a long-lived closure — that would create a mesh-bound reference cycle that segfaults
across in-process runs).
"""

from typing import Any, Literal

import pybFoam as pyf
from pybFoam.meshing import checkMesh, generate_blockmesh, generate_snappy_hex_mesh
from pydantic import BaseModel

from neofoam.framework.initialization import InitStep, InitStepExecutionError, lazy
from neofoam.framework.tools import PreprocessConfig, Tool


class BlockMeshStep(BaseModel):
    """Generate the base mesh in-process from ``blockMeshDict``."""

    tool: Literal["blockMesh"]
    dict_file: str = "system/blockMeshDict"
    verbose: bool = False


class SnappyHexMeshStep(BaseModel):
    """Refine/snap the prior mesh in-process from ``snappyHexMeshDict``."""

    tool: Literal["snappyHexMesh"]
    dict_file: str = "system/snappyHexMeshDict"
    overwrite: bool = True
    verbose: bool = True


class CheckMeshStep(BaseModel):
    """Validate the prior mesh; optionally fail the run on errors."""

    tool: Literal["checkMesh"]
    all_topology: bool = False
    all_geometry: bool = False
    check_quality: bool = False
    fail_on_error: bool = True


blockMeshTool = Tool("blockMesh")
snappyHexMeshTool = Tool("snappyHexMesh")
checkMeshTool = Tool("checkMesh")

# Surface the enable file in configurations() (deduped downstream).
blockMeshTool.config(PreprocessConfig)
snappyHexMeshTool.config(PreprocessConfig)
checkMeshTool.config(PreprocessConfig)


@blockMeshTool.build
def _build_block(cfg: BlockMeshStep) -> list[InitStep]:
    def gen(ctx: dict[str, Any]) -> Any:
        return generate_blockmesh(
            ctx["_foam_time"],
            pyf.dictionary.read(cfg.dict_file),
            verbose=cfg.verbose,
        )

    return [lazy("preprocess.blockMesh", gen, depends_on=["_foam_time"])]


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


@checkMeshTool.build
def _build_check(cfg: CheckMeshStep) -> list[InitStep]:
    def run(ctx: dict[str, Any]) -> Any:
        mesh = ctx["_prev_mesh"]
        try:
            stats = checkMesh(
                mesh,
                all_topology=cfg.all_topology,
                all_geometry=cfg.all_geometry,
                check_quality=cfg.check_quality,
            )
        except (ValueError, TypeError):
            raise
        except Exception as exc:  # binding-level failure
            raise InitStepExecutionError("preprocess.checkMesh", [], exc) from exc
        if cfg.fail_on_error and not stats["passed"]:
            raise InitStepExecutionError(
                "preprocess.checkMesh",
                [],
                RuntimeError(f"mesh check failed: {stats['total_errors']} error(s)"),
            )
        # checkMesh only validates; pass the mesh straight through so the
        # pipeline treats this like any other mesh-advancing step.
        return mesh

    return [InitStep(name="preprocess.checkMesh", initializer=run)]
