# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The script front door: execute a case's ``system/postProcess.py``.

Executing case-local Python is the same trust boundary as running the case at
all, so the loader is a
:class:`~neofoam.core.case_module.CaseModuleLoader` (path confinement, plugin
snapshot/undo) and is called from
:func:`~neofoam.postprocess.table.tables_for_case` only — the one place a
solver run assembles a case's tables, and not part of the package's public
surface, so no MCP, agent or tooling frontend executes a case script by
importing it::

    tables = load_script(case_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from neofoam.core.case_module import CaseModuleLoader, single_instance
from neofoam.postprocess.table import TableSet

#: Where a case keeps its post-processing script.
SCRIPT_FILE = "system/postProcess.py"

#: The only families a re-load may take registrations back from: a script's own
#: nodes, sources and writers. Anything it registers elsewhere (a write policy, a
#: turbulence model) is the process's, not the script's.
_LOADER = CaseModuleLoader(
    module_name="neofoam_case_post",
    families=("Source", "Node", "TableWriter"),
    kind="postProcess",
)


def load_script(case_dir: Path, rel_path: str = SCRIPT_FILE) -> Optional[TableSet]:
    """Execute a case's post-processing script and return its single TableSet.

    ``None`` when the case has no script. Run *before* the spec file is
    resolved, so the ``@Node.register`` classes the script defines are already
    part of the union the spec resolves against. Raises
    :class:`~neofoam.tooling.workspace.CaseAccessError` for a path that leaves
    the case and ``ValueError`` when the script does not define exactly one
    module-level :class:`~neofoam.postprocess.table.TableSet`.
    """
    module = _LOADER.load(case_dir, rel_path)
    if module is None:
        return None
    return single_instance(module, TableSet, "postProcess")
