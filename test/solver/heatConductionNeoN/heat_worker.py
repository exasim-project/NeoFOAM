# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run a heatConductionNeoN case in the working directory; print what it computed as JSON.

``python heat_worker.py single`` runs the single-region case, ``python
heat_worker.py multi`` the ``multiRun`` case. The answer is the single stdout
line prefixed ``#RESULT`` (OpenFOAM and NeoN write freely to stdout): per region
``T``, its time (value, index, deltaT), each coupled patch's face centres,
wall temperature and face values of ``T``, and the passes of the leader's
coupling per time step.

Every message of the multiRun comm_channel is pickled, as between processes
(``InProcessCommChannel.PICKLE``), so a message that is not plain data fails.

A subprocess: Kokkos and ``Foam::Time`` are per-process, and NeoN must be
finalized after every NeoN object is gone, so the contexts never outlive
:func:`main`.
"""

import json
import sys
from typing import Any

import numpy as np

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling, InProcessCommChannel
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.solver.heatConductionNeoN import run as run_single
from neofoam.solver.multiRun import run as run_multi

RESULT_PREFIX = "#RESULT "


def _host(vector: Any) -> list[Any]:
    return np.asarray(vector.copy_to_host()).tolist()


def _region(ctx: Context) -> dict[str, Any]:
    return {
        "T": _host(ctx.fields["T"].internal_vector()),
        "time": {"value": ctx.time.value, "index": ctx.time.index, "deltaT": ctx.time.delta_t},
        "patches": {
            patch.patch: {
                "type": type(patch).__name__,
                "points": np.asarray(patch.points()).tolist(),
                "wall_T": np.asarray(patch.wall_T()).tolist(),
                "face_T": _host(ctx.fields["T"].boundary_data_value())[patch.faces],
            }
            for patch in ctx.models["conduction"].patches
        },
    }


def main(mode: str) -> dict[str, Any]:
    if mode == "single":
        return {"regions": {"case": _region(run_single(["heatConductionNeoN"]))}}
    InProcessCommChannel.PICKLE = True
    regions = run_multi()
    passes: dict[str, list[int]] = {}
    for name, ctx in regions.items():
        operations = Coupling.of(ctx).operations()
        op = operations[0].func if len(operations) else None
        if isinstance(op, CouplingOp) and op.controller is not None:
            passes[name] = [outcome.passes for outcome in op.controller.outcomes]
    return {"regions": {name: _region(ctx) for name, ctx in regions.items()}, "passes": passes}


if __name__ == "__main__":
    print(RESULT_PREFIX + json.dumps(main(sys.argv[1])), flush=True)
