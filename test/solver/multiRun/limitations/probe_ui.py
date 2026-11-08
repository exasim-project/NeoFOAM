# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The limitations cases in the neofoam_ui DAG stepper: ``--solver probe_ui:probe``.

The worker needs this directory on its PYTHONPATH (neofoam_ui's ``scripts/probe.sh`` sets
it) and runs in the case directory.
"""

import sys
from pathlib import Path
from typing import Any

# test/, for solver.multiRun.limitations; last, so its packages shadow nothing installed
sys.path.append(str(Path(__file__).resolve().parents[3]))


def probe() -> Any:
    """multiRun of the case in the working directory, with the ``probe`` region solver."""
    from neofoam_ui.adapters.neofoam import ObservedNeoFoamSolver  # noqa: PLC0415

    from solver.multiRun.limitations import run  # noqa: PLC0415

    run.use_probe()
    # multiRun.run() takes no argv
    return ObservedNeoFoamSolver("multiRun", lambda argv: run.MULTI_RUN.run())
