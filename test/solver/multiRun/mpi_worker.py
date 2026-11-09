# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the multiRun case in the working directory; print each region's ``T`` as JSON.

``python mpi_worker.py`` runs it on threads, ``mpirun -np <n> python
mpi_worker.py`` on processes. Every process prints one stdout line prefixed
``#RESULT`` (OpenFOAM writes freely to stdout): per region it ran, the cell
centres and ``T`` of its cells, with the global rank; for a ``foamDiffusion``
region from OpenFOAM, for a ``heatConductionNeoN`` region the centres from its
OpenFOAM mesh and ``T`` from NeoN. multiRun finds region
solvers by package name; the worker makes it find the test solver
``foamDiffusion`` (:mod:`solver.multiRun.foam_solver`) as well.

A coupled region also prints, per coupled side on this process, the side's
points and the :data:`~neofoam.framework.coupling.Array` data it sends at the
end (``sides``, by ``<region>.<patch>``): the interface, e.g. the wall
temperature of ``coupledBlocks``' left side.

``python mpi_worker.py fail:<region>`` makes ``<region>`` raise in its second
time step (``foamDiffusion``); ``fail-coupled:<region>:<rank>`` makes the
coupled sides of ``<region>`` raise on global rank ``<rank>`` as they send for
the third time (any coupled solver).
"""

import importlib
import json
import os
import sys
from typing import Any

import numpy as np

from neofoam.framework.coupling import data as coupling_data
from neofoam.framework.coupling import loop as coupling_loop
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from solver.multiRun.foam_solver import FAIL_VARIABLE, foamDiffusion

RESULT_PREFIX = "#RESULT "
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")


def _resolve(name: str) -> Any:
    return foamDiffusion if name == "foamDiffusion" else resolve_solver(name)


def _cells(ctx: Any) -> dict[str, Any]:
    """The centres and ``T`` of the region's cells on this process."""
    if "foam" in ctx.models:
        mesh, temperature = ctx.models["foam"].mesh, ctx.fields["T"].internalField()
    else:
        mesh = ctx.models["neon_runtime"].mesh
        temperature = ctx.fields["T"].internal_vector().copy_to_host()
    return {
        "centres": np.asarray(mesh.C().internalField()).tolist(),
        "T": np.asarray(temperature).tolist(),
        "index": ctx.time.index,
        "sides": _sides(ctx),
    }


def _sides(ctx: Any) -> dict[str, Any]:
    """Per coupled side of the region on this process: its points and the Array data it sends."""
    coupling = ctx.models.get("coupling")
    if coupling is None or coupling.coupling is None:
        return {}
    sides = {}
    for side in coupling.sides:
        sent = coupling_loop._sent(side)
        sides[side.label] = {
            "points": np.asarray(side.coordinates).reshape(-1, 3).tolist(),
            **{
                name: np.asarray(sent[name]).tolist()
                for name in coupling_data._array_names(side.sends)
            },
        }
    return sides


def _fail_coupled(region: str, rank: int) -> None:
    """Make the coupled sides of ``region`` raise on global rank ``rank`` at their third send."""
    sent = coupling_loop._sent
    calls = [0]

    def failing(side: Any) -> Any:
        if side.region == region and int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)) == rank:
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError(f"region {region} fails on rank {rank} as asked")
        return sent(side)

    coupling_loop._sent = failing


def main(argv: list[str]) -> dict[str, Any]:
    MULTI_RUN_MODULE.resolve_solver = _resolve  # type: ignore[attr-defined]
    failing = next((arg.split(":", 1)[1] for arg in argv if arg.startswith("fail:")), None)
    if failing is not None:
        os.environ[FAIL_VARIABLE] = failing
    coupled = next((arg.split(":")[1:] for arg in argv if arg.startswith("fail-coupled:")), None)
    if coupled is not None:
        _fail_coupled(coupled[0], int(coupled[1]))
    regions = run()
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    return {
        "rank": rank,
        "regions": {region: _cells(ctx) for region, ctx in regions.items()},
    }


if __name__ == "__main__":
    print(RESULT_PREFIX + json.dumps(main(sys.argv[1:])), flush=True)
