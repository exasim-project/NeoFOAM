# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Read a materialized case's field back into numpy, one subprocess per read.

Kept off the :class:`~neofoam.tooling.casebuild.pipeline.CaseDir` value (which is a
pure case-*construction* handle) so ``pipeline.py`` stays free of numpy/subprocess:
reading a field is a distinct concern from building a case.

Constructing ``Foam::Time`` twice in one interpreter corrupts OpenFOAM per-process
global state (later reads come back as ``nan``), so each read runs in its own process
via the ``neofoam.tooling.casebuild._reader`` entry point — the only safe way to read
more than one field from Python.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neofoam.tooling.casebuild.pipeline import CaseDir

if TYPE_CHECKING:
    import numpy as np


def read_field(
    case: CaseDir, name: str, *, time: str = "latest"
) -> "np.ndarray[Any, Any]":
    """Read field *name*'s internal field from *case* (``time`` = ``"latest"`` or a dir).

    Runs the read in a fresh subprocess (see the module docstring). Returns the
    internal field as a numpy array (``(N,)`` scalar, ``(N, 3)`` vector).
    """
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{name}.npy"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "neofoam.tooling.casebuild._reader",
                str(case.path),
                time,
                name,
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return np.asarray(np.load(out))
