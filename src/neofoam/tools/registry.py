# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Process-wide registry of preprocessing tools.

A tool module self-registers via :func:`register_tool` at import; importing the
``neofoam.tools`` package imports every built-in tool module, so :func:`available_tools`
returns them all. Resolution (``resolve_tools``) is scoped to this set — adding a tool is
a new self-registering module, with no edit to any solver or the CLI.
"""

from neofoam.framework.tools.spec import ToolSpec

_REGISTRY: dict[str, ToolSpec] = {}


def register_tool(tool: ToolSpec) -> ToolSpec:
    """Register ``tool`` under its name (idempotent); return it for module-level use."""
    _REGISTRY[tool.name] = tool
    return tool


def available_tools() -> list[ToolSpec]:
    """Every registered tool (registry populated by importing ``neofoam.tools``)."""
    return list(_REGISTRY.values())
