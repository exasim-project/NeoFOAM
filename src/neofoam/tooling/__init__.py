# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``neofoam.tooling`` — frontend infrastructure above the library.

Thin, dependency-light package init: it re-exports only the workspace sandbox so
``import neofoam.tooling`` stays stdlib-only (no trame / fastmcp / pybFoam). Heavier
submodules (mcp host, ui, fill) import their own extras lazily.

Interface (``__all__`` — the whole exported surface of *this* package):

* :class:`~neofoam.tooling.workspace.Workspace` — confine an untrusted ``case_id``
  under a fixed root directory (the frontend trust boundary).
* :class:`~neofoam.tooling.workspace.CaseAccessError` — the typed ``ValueError``
  raised when a path escapes that root.

The two heavier sub-packages are **deliberately not** re-exported here — importing
them pulls pybFoam / PyYAML and would break the stdlib-only guarantee above. Reach
them by their own paths, each of which carries its own narrow interface:
:mod:`neofoam.tooling.casebuild` (pipe-composed case construction) and
:mod:`neofoam.tooling.workflow` (a directory of deep sub-modules — ``geometry``
for geometry → mesh, ``sweep`` for parameter sweeps, ``rules``, ``dag``).
"""

from neofoam.tooling.workspace import CaseAccessError, Workspace

__all__ = ["CaseAccessError", "Workspace"]
