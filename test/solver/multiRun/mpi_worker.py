# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the multiRun case in the working directory; print each region's ``T`` as JSON.

``python mpi_worker.py`` runs it on threads, ``mpirun -np <n> python
mpi_worker.py`` on processes. Every process prints one stdout line prefixed
``#RESULT`` (OpenFOAM writes freely to stdout): per region it ran, the cell
centres and ``T`` of its cells, with the global rank. multiRun finds region
solvers by package name; the worker makes it find the test solver
``foamDiffusion`` (:mod:`solver.multiRun.foam_solver`) as well.

``python mpi_worker.py fail:<region>`` makes ``<region>`` raise in its second
time step.
"""

import importlib
import json
import os
import sys
from typing import Any

import numpy as np

from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from solver.multiRun.foam_solver import FAIL_VARIABLE, foamDiffusion

RESULT_PREFIX = "#RESULT "
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")


def _resolve(name: str) -> Any:
    return foamDiffusion if name == "foamDiffusion" else resolve_solver(name)


def main(argv: list[str]) -> dict[str, Any]:
    MULTI_RUN_MODULE.resolve_solver = _resolve  # type: ignore[attr-defined]
    failing = next((arg.split(":", 1)[1] for arg in argv if arg.startswith("fail:")), None)
    if failing is not None:
        os.environ[FAIL_VARIABLE] = failing
    regions = run()
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    return {
        "rank": rank,
        "regions": {
            region: {
                "centres": np.asarray(ctx.models["foam"].mesh.C().internalField()).tolist(),
                "T": np.asarray(ctx.fields["T"].internalField()).tolist(),
                "index": ctx.time.index,
            }
            for region, ctx in regions.items()
        },
    }


if __name__ == "__main__":
    print(RESULT_PREFIX + json.dumps(main(sys.argv[1:])), flush=True)
