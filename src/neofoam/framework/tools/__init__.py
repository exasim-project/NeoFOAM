# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""neofoam.framework.tools — shared preprocessing Tool abstraction."""

from .pipeline import PreprocessConfig, resolve_pipeline, tool_init_steps
from .spec import Tool, ToolRuntime, ToolSpec

__all__ = [
    "ToolSpec",
    "Tool",
    "ToolRuntime",
    "PreprocessConfig",
    "resolve_pipeline",
    "tool_init_steps",
]
