# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run ``incompressibleVoF`` to completion in one already-prepared case.

``run(...)`` constructs a ``Foam::Time``, and a process may own exactly one —
so every regime variant of ``test_mules_regimes.py`` gets its own interpreter
through this worker. Everything the test asserts on is what the solver writes
to the case's time directories; the worker itself produces no artifacts.

Usage: ``python _mules_regime_worker.py <case-dir>``
"""

import os
import sys
from pathlib import Path

from neofoam.solver.incompressibleVoF import run

if __name__ == "__main__":
    os.chdir(Path(sys.argv[1]))
    run(["incompressibleVoF"])
