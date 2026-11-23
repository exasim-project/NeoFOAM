# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``mpirun`` of the VTKHDF tests' MPI, and the mark of tests that need it with mpi4py."""

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest


def mpirun() -> Optional[str]:
    """The ``mpirun`` next to the interpreter (its environment's MPI), else the one on PATH."""
    beside = Path(sys.executable).parent / "mpirun"
    return str(beside) if beside.exists() else shutil.which("mpirun")


_mpi_missing = importlib.util.find_spec("mpi4py") is None or mpirun() is None
# A CI job that sets this must run the tests: without it, a broken setup would only skip them.
if _mpi_missing and os.environ.get("NEOFOAM_REQUIRE_MPI_TESTS"):
    raise RuntimeError("NEOFOAM_REQUIRE_MPI_TESTS is set, but mpi4py or mpirun is missing")

needs_mpi = pytest.mark.skipif(_mpi_missing, reason="needs mpi4py and mpirun")
