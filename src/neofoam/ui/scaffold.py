# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Write OpenFOAM-style ``Allrun`` / ``Allclean`` scripts into a saved case.

After the configs are written, this makes the case directory *runnable*: ``./Allrun``
launches the NeoFOAM ``incompressibleFluid`` solver (meshing happens in-process from
``system/preprocess.yaml``), and ``./Allclean`` restores it to a clean state. ``Allrun``
reuses the repo's ``scripts/Allrun`` template when available (source checkout), falling
back to an embedded copy so an installed package still works; ``Allclean`` is authored
here (the repo's ``scripts/Allclean`` is empty).
"""

from __future__ import annotations

import stat
from pathlib import Path

__all__ = ["scaffold_runnable_case", "allrun_template_path", "ALLCLEAN_TEXT"]

# Embedded copy of scripts/Allrun — kept in sync; used when the source-tree template
# is absent (e.g. an installed wheel).
_ALLRUN_FALLBACK = """\
#!/usr/bin/env bash
#
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# OpenFOAM-convention Allrun for a NeoFOAM incompressibleFluid case.
#
# The single solver command does everything: meshing (blockMesh / snappyHexMesh,
# driven in-process by system/preprocess.yaml) followed by the PIMPLE solve.
#
# Usage (from the case directory):
#   ./Allrun [extra solver args…]
#   NEOFOAM_PYTHON=/path/to/python ./Allrun     # pin the interpreter
#
cd "${0%/*}" || exit 1   # run from this case directory

if [ -n "${NEOFOAM_PYTHON:-}" ]; then
    exec "$NEOFOAM_PYTHON" -m neofoam.cli.app solver incompressiblefluid "$@"
fi
exec neofoam solver incompressiblefluid "$@"
"""

ALLCLEAN_TEXT = """\
#!/bin/sh
#
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# OpenFOAM-convention Allclean for a NeoFOAM incompressibleFluid case.
#
cd "${0%/*}" || exit 1
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions
cleanCase0
"""

_EXEC_MODE = (
    stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
)  # 0o755


def allrun_template_path() -> Path:
    """Locate the repo's ``scripts/Allrun`` (source checkout)."""
    return Path(__file__).resolve().parents[3] / "scripts" / "Allrun"


def _allrun_text() -> str:
    template = allrun_template_path()
    if template.is_file():
        return template.read_text()
    return _ALLRUN_FALLBACK


def scaffold_runnable_case(case_dir: Path | str) -> list[Path]:
    """Write executable ``Allrun`` + ``Allclean`` into ``case_dir``. Idempotent."""
    case = Path(case_dir)
    written: list[Path] = []
    for name, text in (("Allrun", _allrun_text()), ("Allclean", ALLCLEAN_TEXT)):
        path = case / name
        path.write_text(text)
        path.chmod(_EXEC_MODE)
        written.append(path)
    return written
