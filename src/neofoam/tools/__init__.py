# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""neofoam.tools — shared, solver-agnostic preprocessing tools.

Importing this package self-registers every built-in tool into the shared registry.
"""

from . import block_mesh, check_mesh, snappy_hex_mesh  # noqa: F401  (populate registry)
from .block_mesh import (
    Block,
    BlockMeshDictConfig,
    BlockMeshStep,
    BlockPatch,
    blockMeshTool,
)
from .check_mesh import CheckMeshStep, checkMeshTool
from .registry import available_tools, register_tool
from .snappy_hex_mesh import (
    SnappyHexMeshDictConfig,
    SnappyHexMeshStep,
    SnappySurface,
    snappyHexMeshTool,
)

__all__ = [
    "available_tools",
    "register_tool",
    "blockMeshTool",
    "snappyHexMeshTool",
    "checkMeshTool",
    "BlockMeshStep",
    "SnappyHexMeshStep",
    "CheckMeshStep",
    "BlockMeshDictConfig",
    "SnappyHexMeshDictConfig",
    "Block",
    "BlockPatch",
    "SnappySurface",
]
