# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the multiRun case in the working directory with ``regions_recorder``; print its calls.

Under ``mpirun``: one ``#RESULT`` JSON line per process, its region and the calls.
"""

import importlib
import json

from neofoam.solver.multiRun import run
from solver.multiRun.conftest import SOLVERS
from solver.multiRun.insitu_recorder import CALLS

MULTI_RUN = importlib.import_module("neofoam.solver.multiRun.multiRun")
_resolve = MULTI_RUN.resolve_solver
MULTI_RUN.resolve_solver = lambda name: SOLVERS.get(name) or _resolve(name)

contexts = run()
for region in contexts:
    print("#RESULT " + json.dumps({"region": region, "calls": CALLS}), flush=True)
