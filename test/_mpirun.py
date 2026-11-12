# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``mpirun`` of the tests' MPI, and the mark of tests that need it with mpi4py."""

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest


def mpirun() -> Optional[str]:
    """The ``mpirun`` next to the interpreter (its environment's MPI), else the one on PATH."""
    beside = Path(sys.executable).parent / "mpirun"
    return str(beside) if beside.exists() else shutil.which("mpirun")


needs_mpi = pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None or mpirun() is None,
    reason="needs mpi4py and mpirun (the extra neofoam[mpi])",
)
