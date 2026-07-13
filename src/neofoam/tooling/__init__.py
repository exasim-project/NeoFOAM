# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neofoam.tooling`` — frontend infrastructure above the library.

Thin, dependency-light package init: it re-exports only the workspace sandbox so
``import neofoam.tooling`` stays stdlib-only (no trame / fastmcp / pybFoam). Heavier
submodules (mcp host, ui, fill) import their own extras lazily.
"""

from neofoam.tooling.workspace import CaseAccessError, Workspace

__all__ = ["CaseAccessError", "Workspace"]
