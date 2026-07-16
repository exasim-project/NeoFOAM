# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared context manager for managing working directory during solver runs."""

import contextlib
import os
from pathlib import Path


@contextlib.contextmanager
def cwd(path: Path):
    """Context manager to temporarily change working directory.

    Usage:
        with cwd(case.path):
            run(["incompressibleFluid"])
    """
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)
