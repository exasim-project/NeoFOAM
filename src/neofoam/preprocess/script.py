# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The script front door: execute a case's ``system/setFields.py``.

Executing case-local Python is the same trust boundary as running the case at
all, so the loader is a :class:`~neofoam.core.case_module.CaseModuleLoader` (path
confinement, plugin snapshot/undo) and :func:`set_fields_for_case` — the one
place the ``setFields`` tool assembles a case's assignments — is the only caller,
so no MCP, agent or tooling frontend executes a case script by importing the
package::

    setup = set_fields_for_case(case_dir)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from neofoam.core.case_module import CaseModuleLoader, single_instance
from neofoam.postprocess.nodes.selectors import Selector
from neofoam.preprocess.config import (
    SPEC_FILES,
    RegionValue,
    load_config,
    resolve_regions,
    spec_file,
)

#: Where a case keeps its field-initialisation script.
SCRIPT_FILE = "system/setFields.py"

#: A ``setFields`` script may register its own selector; nothing else of what it
#: imports is the script's to unregister.
_LOADER = CaseModuleLoader(
    module_name="neofoam_case_set_fields",
    families=("Node",),
    kind="setFields",
)


class SetFields:
    """What one case initialises: a default per field, then a value per region.

    A case script creates exactly one at module level and calls :meth:`assign` for
    every region, building the regions with the *post-processing* selectors so
    there is one region language::

        from neofoam.postprocess import Box, Sphere
        from neofoam.preprocess import SetFields

        setFields = SetFields(defaults={"alpha.water": 0.0})
        setFields.assign(Box(min=(0, 0, -1), max=(0.1461, 0.292, 1)), {"alpha.water": 1.0})

    It is a plain declaration — nothing is read or written until the ``setFields``
    tool applies it to a mesh.
    """

    def __init__(self, defaults: Optional[Mapping[str, RegionValue]] = None) -> None:
        self.defaults: dict[str, RegionValue] = dict(defaults or {})
        self.regions: list[tuple[Selector, dict[str, RegionValue]]] = []

    def assign(self, region: Selector, values: Mapping[str, RegionValue]) -> None:
        """Set every field of ``values`` inside ``region``; later calls win."""
        self.regions.append((region, dict(values)))


def load_script(case_dir: Path, rel_path: str = SCRIPT_FILE) -> Optional[SetFields]:
    """Execute a case's setFields script and return its single SetFields.

    ``None`` when the case has no script. Run *before* the spec file is resolved,
    so the ``@Node.register`` classes the script defines are already part of the
    union the spec resolves against. Raises
    :class:`~neofoam.tooling.workspace.CaseAccessError` for a path that leaves the
    case and ``ValueError`` when the script does not define exactly one
    module-level :class:`SetFields`.
    """
    module = _LOADER.load(case_dir, rel_path)
    if module is None:
        return None
    return single_instance(module, SetFields, "setFields")


def set_fields_for_case(case_dir: Path) -> SetFields:
    """Everything a case declares: the script's regions first, then the spec file's.

    The script runs first so the selectors it registers with ``@Node.register`` are
    selectable by ``type`` from the spec file, and a default declared by both is
    the spec file's. A case that declares neither front door raises — the tool was
    listed in ``system/preprocess.yaml`` with nothing to do.
    """
    case_dir = Path(case_dir)
    script = load_script(case_dir)
    path = spec_file(case_dir)
    if script is None and path is None:
        raise ValueError(
            f"setFields: {case_dir} declares neither {SCRIPT_FILE} nor one of "
            f"{list(SPEC_FILES)} — remove the tool from system/preprocess.yaml "
            f"or declare what to set"
        )

    setup = script if script is not None else SetFields()
    if path is not None:
        config = load_config(case_dir)
        setup.defaults.update(config.defaults)
        setup.regions.extend(resolve_regions(config))
    return setup
