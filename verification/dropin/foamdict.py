# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Static readers for OpenFOAM case dictionaries — classify without running.

A study's ``discover.py`` uses these to tier the tutorial tree in milliseconds,
so the full inventory is known before a single solver starts. Deliberately
regex-based rather than routed through ``neofoam.io.DictFile``: this must read
*any* tutorial's dictionaries, including ones that fail to parse or ship non-UTF8
bytes, and answer "absent" instead of raising.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = [
    "entries",
    "entry",
    "read",
    "turbulence",
    "tutorials_root",
    "uses_ami",
]


def tutorials_root(group: str) -> Path | None:
    """``$FOAM_TUTORIALS/<group>``, or ``None`` when no OpenFOAM is sourced."""
    root = os.environ.get("FOAM_TUTORIALS")
    if not root:
        project = os.environ.get("WM_PROJECT_DIR")
        root = f"{project}/tutorials" if project else None
    if not root:
        return None
    path = Path(root) / group
    return path if path.is_dir() else None


def read(path: Path) -> str:
    """Read a case dictionary leniently — some tutorials ship non-UTF8 bytes."""
    if not path.is_file():
        return ""
    return path.read_bytes().decode("utf-8", "replace")


def entry(text: str, key: str) -> str:
    """The value of a scalar dict entry; ``""`` when absent."""
    match = re.search(rf"^\s*{key}\s+([^;]+);", text, re.MULTILINE)
    return match.group(1).strip().strip('"') if match else ""


def entries(text: str, key: str) -> list[str]:
    """Every value for *key* — a dictionary may set it once per sub-dict.

    Needed wherever a single reading would be wrong: ``waveMakerFlap`` and
    ``waveMakerPiston`` both declare ``nAlphaSubCycles`` twice, and a classifier
    that took the first would tier them on the wrong number.
    """
    return [
        match.group(1).strip().strip('"')
        for match in re.finditer(rf"^\s*{key}\s+([^;]+);", text, re.MULTILINE)
    ]


def turbulence(case: Path) -> tuple[str, str]:
    """``(simulationType, model)`` from momentumTransport/turbulenceProperties."""
    for name in ("momentumTransport", "turbulenceProperties"):
        text = read(case / "constant" / name)
        if not text:
            continue
        simulation = entry(text, "simulationType")
        if simulation == "laminar" or not simulation:
            return (simulation or "laminar", "laminar")
        model = entry(text, "RASModel") or entry(text, "LESModel")
        return simulation, model or "unknown"
    return "laminar", "laminar"


def uses_ami(case: Path) -> bool:
    """True when the case couples patches through a cyclicAMI interface."""
    if "cyclicAMI" in read(case / "constant" / "polyMesh" / "boundary"):
        return True
    # Most tutorials only grow polyMesh at run time, so the intent lives in the
    # mesh-generation dictionaries and the 0/ boundary conditions instead.
    for sub in ("system", "0", "0.orig", "constant"):
        directory = case / sub
        if not directory.is_dir():
            continue
        for item in directory.rglob("*"):
            if item.is_file() and "cyclicAMI" in read(item):
                return True
    return False
