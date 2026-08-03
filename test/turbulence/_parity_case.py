# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Staging and subprocess plumbing shared by the turbulence parity tests.

The parity cases differ from one another in a handful of files (a seeded ``0/k``,
a ``constant/turbulenceProperties``), so only one full case per mesh is checked in
and each variant ships just its differing files. :func:`stage` composes them into
``tmp_path``; :func:`run_worker` drives :mod:`_parity_worker`, one ``Foam::Time``
per process.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_WORKER = Path(__file__).parent / "_parity_worker.py"


def stage(base: Path, dest: Path, *overlays: Path) -> Path:
    """Copy the full case *base* into *dest*, then lay each variant tree on top."""
    shutil.copytree(base, dest)
    for overlay in overlays:
        shutil.copytree(overlay, dest, dirs_exist_ok=True)
    return dest


def run_worker(role: str, case: Path) -> None:
    """Run one worker role in a fresh process (one ``Foam::Time`` per process)."""
    subprocess.run(
        [sys.executable, str(_WORKER), role, str(case)],
        check=True,
        cwd=str(case.parent),
    )
