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

import shlex
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


def _neo_ico_foam_binary() -> Path | None:
    """The compiled ``neoIcoFoam`` app shipped alongside this package, if present."""
    candidate = Path(__file__).resolve().parents[1] / "bin" / "neoIcoFoam"
    return candidate if candidate.is_file() else None


def _uses_piso(case_dir: Path) -> bool:
    """True when ``system/fvSolution`` has a ``PISO`` block and no ``PIMPLE`` block.

    Marks a classic OpenFOAM case (e.g. one imported as-is because it didn't fit
    incompressiblefluid's PIMPLE schema) rather than one authored by the wizard.
    """
    path = case_dir / "system" / "fvSolution"
    if not path.is_file():
        return False
    try:
        import pybFoam as pyf

        root = pyf.dictionary.read(str(path))
        return bool(root.found("PISO")) and not root.found("PIMPLE")
    except Exception:  # noqa: BLE001 - unreadable/malformed dict; not a PISO case
        return False


def _neo_ico_foam_allrun_text(binary: Path) -> str:
    """Allrun for a PISO-format case: mesh with blockMesh, then run neoIcoFoam."""
    return f"""\
#!/usr/bin/env bash
#
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# This case's system/fvSolution uses a classic PISO sub-dict rather than
# incompressiblefluid's PIMPLE format (e.g. imported from an existing OpenFOAM
# case as-is), so it runs through the compiled neoIcoFoam solver instead, which
# needs a pre-generated mesh.
#
# Usage (from the case directory):
#   ./Allrun [extra solver args…]
#
cd "${{0%/*}}" || exit 1   # run from this case directory

blockMesh || exit 1
exec {shlex.quote(str(binary))} "$@"
"""


def scaffold_runnable_case(case_dir: Path | str) -> list[Path]:
    """Write executable ``Allrun`` + ``Allclean`` into ``case_dir``. Idempotent.

    ``Allrun`` launches the compiled ``neoIcoFoam`` solver (after ``blockMesh``)
    when the case's ``system/fvSolution`` is PISO-format rather than PIMPLE (see
    :func:`_uses_piso`) and that binary is available; otherwise the usual
    ``incompressiblefluid`` launcher is used.
    """
    case = Path(case_dir)
    binary = _neo_ico_foam_binary()
    allrun_text = (
        _neo_ico_foam_allrun_text(binary)
        if binary is not None and _uses_piso(case)
        else _allrun_text()
    )
    written: list[Path] = []
    for name, text in (("Allrun", allrun_text), ("Allclean", ALLCLEAN_TEXT)):
        path = case / name
        path.write_text(text)
        path.chmod(_EXEC_MODE)
        written.append(path)
    return written
