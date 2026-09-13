# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""checkMesh tool: validate the prior mesh in-process and optionally fail the run.

The ``@build`` initializer resolves the prior mesh from the live ``ctx`` at call time
(never captured in a long-lived closure — that would create a mesh-bound reference cycle
that segfaults across in-process runs). ``checkMesh`` is a module global so tests can
monkeypatch it without building a mesh. checkMesh only validates; on success it passes
the prior mesh straight through so the pipeline treats it like any mesh-advancing step.
"""

from typing import Any, Literal

from pybFoam.meshing import checkMesh
from pydantic import BaseModel

from neofoam.framework.initialization import InitStep, InitStepExecutionError
from neofoam.framework.tools import Tool

from .registry import register_tool


class CheckMeshStep(BaseModel):
    """Validate the prior mesh; optionally fail the run on errors."""

    tool: Literal["checkMesh"]
    all_topology: bool = False
    all_geometry: bool = False
    check_quality: bool = False
    fail_on_error: bool = True


checkMeshTool = Tool("checkMesh")


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


register_tool(checkMeshTool)
