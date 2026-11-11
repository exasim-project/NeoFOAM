# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the multiRun case in the working directory; print a ``#RESULT`` JSON line per process.

Per region on this process: the cell centres, ``T``, the time index and, per
coupled port, its points and what it sends. ``python mpi_worker.py`` runs on
threads, under ``mpirun`` on processes. ``fail:<region>`` makes a
``foamDiffusion`` region raise; ``fail-coupled:<region>:<rank>`` makes the
ports of ``<region>`` raise on global rank ``<rank>`` (a follower) at their
third send.
"""

import importlib
import json
import os
import sys
from typing import Any

import numpy as np

from neofoam.framework.coupling import data as coupling_data
from neofoam.framework.coupling.comm import mpi as coupling_mpi
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
        "sides": _ports(ctx),
    }


def _ports(ctx: Any) -> dict[str, Any]:
    """Per coupled port of the region on this process: its points and the Array data it sends."""
    coupling = ctx.models.get("coupling")
    if coupling is None or coupling.coupling is None:
        return {}
    ports = {}
    for port in coupling.ports:
        points = port.location.geometry(coupling.geometry).points
        sent = {
            name: values
            for stream in port.sends
            for name, values in coupling_data.sent(stream, port, "", (0, None)).items()
        }
        ports[port.label] = {
            "points": np.asarray(points).reshape(-1, 3).tolist(),
            **{name: np.asarray(values).tolist() for name, values in sent.items()},
        }
    return ports


def _fail_coupled(region: str, rank: int) -> None:
    """Make ``region``'s ports raise on global rank ``rank`` (a follower) at their third send."""
    sent = coupling_mpi.sent
    calls = [0]

    def failing(stream: Any, port: Any, *args: Any) -> Any:
        if port.region == region and int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)) == rank:
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError(f"region {region} fails on rank {rank} as asked")
        return sent(stream, port, *args)

    coupling_mpi.sent = failing  # type: ignore[assignment]


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
