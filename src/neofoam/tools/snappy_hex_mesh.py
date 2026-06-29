# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""snappyHexMesh tool: refine/snap the prior mesh in-process from ``snappyHexMeshDict``.

The ``@build`` initializer resolves the prior mesh from the live ``ctx`` at call time
(never captured in a long-lived closure — that would create a mesh-bound reference cycle
that segfaults across in-process runs). ``pyf`` / ``generate_snappy_hex_mesh`` are module
globals so tests can monkeypatch them without building a mesh.
"""

from typing import Any, Literal

import pybFoam as pyf
from pybFoam.meshing import generate_snappy_hex_mesh
from pydantic import BaseModel

from neofoam.framework.initialization import InitStep, lazy
from neofoam.framework.tools import Tool

from .registry import register_tool


class SnappyHexMeshStep(BaseModel):
    """Refine/snap the prior mesh in-process from ``snappyHexMeshDict``."""

    tool: Literal["snappyHexMesh"]
    dict_file: str = "system/snappyHexMeshDict"
    overwrite: bool = True
    verbose: bool = True


snappyHexMeshTool = Tool("snappyHexMesh")


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
