# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Write OpenFOAM-style ``Allrun`` / ``Allclean`` scripts into a saved case.

After the configs are written, this makes the case directory *runnable*: ``./Allrun``
launches the case's NeoFOAM solver (meshing happens in-process from
``system/preprocess.yaml``), and ``./Allclean`` restores it to a clean state. ``Allrun``
reuses the repo's ``scripts/Allrun`` template when available (source checkout), falling
back to an embedded copy so an installed package still works; ``Allclean`` is authored
here (the repo has no ``scripts/Allclean``).
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
# Copied into the case directory and run from there (``./Allrun``), like an
# OpenFOAM tutorial. The single solver command does everything: meshing
# (blockMesh / snappyHexMesh, driven in-process by system/preprocess.yaml)
# followed by the PIMPLE solve — so this script "just" runs the solver.
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

#: The solver the ``Allrun`` template is written for.
_TEMPLATE_SOLVER = "incompressibleFluid"

_EXEC_MODE = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH  # 0o755


def allrun_template_path() -> Path:
    """Locate the repo's ``scripts/Allrun`` (source checkout)."""
    return Path(__file__).resolve().parents[3] / "scripts" / "Allrun"


def _allrun_text(solver_name: str) -> str:
    template = allrun_template_path()
    text = template.read_text() if template.is_file() else _ALLRUN_FALLBACK
    # The CLI registers each solver under its lower-cased name (neofoam.cli.app).
    text = text.replace(f"NeoFOAM {_TEMPLATE_SOLVER} case", f"NeoFOAM {solver_name} case")
    return text.replace(f"solver {_TEMPLATE_SOLVER.lower()} ", f"solver {solver_name.lower()} ")


def scaffold_runnable_case(case_dir: Path | str, solver_name: str = _TEMPLATE_SOLVER) -> list[Path]:
    """Write executable ``Allrun`` + ``Allclean`` into ``case_dir``. Idempotent.

    ``Allrun`` runs the ``neofoam solver`` command of ``solver_name``.
    """
    case = Path(case_dir)
    written: list[Path] = []
    for name, text in (("Allrun", _allrun_text(solver_name)), ("Allclean", ALLCLEAN_TEXT)):
        path = case / name
        path.write_text(text)
        path.chmod(_EXEC_MODE)
        written.append(path)
    return written
