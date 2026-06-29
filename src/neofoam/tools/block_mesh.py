# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""blockMesh tool: generate the base mesh in-process from ``blockMeshDict``.

The ``@build`` initializer resolves the pybFoam ``Time`` from the live ``ctx`` at call
time (never captured in a long-lived closure — that would create a mesh-bound reference
cycle that segfaults across in-process runs). ``pyf`` / ``generate_blockmesh`` are module
globals so tests can monkeypatch them without building a mesh.
"""

from typing import Any, Literal

import pybFoam as pyf
from pybFoam.meshing import generate_blockmesh
from pydantic import BaseModel

from neofoam.framework.initialization import InitStep, lazy
from neofoam.framework.tools import Tool

from .registry import register_tool


class BlockMeshStep(BaseModel):
    """Generate the base mesh in-process from ``blockMeshDict``."""

    tool: Literal["blockMesh"]
    dict_file: str = "system/blockMeshDict"
    verbose: bool = False


blockMeshTool = Tool("blockMesh")


@blockMeshTool.build
def _build_block(cfg: BlockMeshStep) -> list[InitStep]:
    def gen(ctx: dict[str, Any]) -> Any:
        return generate_blockmesh(
            ctx["_foam_time"],
            pyf.dictionary.read(cfg.dict_file),
            verbose=cfg.verbose,
        )

    return [lazy("preprocess.blockMesh", gen, depends_on=["_foam_time"])]


register_tool(blockMeshTool)
