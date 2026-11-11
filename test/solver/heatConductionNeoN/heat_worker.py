# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run a heatConductionNeoN case (``single|multi``), print the ``#RESULT`` JSON line.

A subprocess: Kokkos and ``Foam::Time`` are per process, and NeoN must be finalized
after its objects are gone.
"""

import json
import sys
from typing import Any

import numpy as np

from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.solver.heatConductionNeoN import run as run_single
from neofoam.solver.multiRun import run as run_multi

RESULT_PREFIX = "#RESULT "


def _host(vector: Any) -> Any:
    return np.asarray(vector.copy_to_host()).tolist()


def _region(ctx: Context) -> dict[str, Any]:
    # T as written at the last time: pybFoam binds no boundaryField()
    written = _host(
        nfb.read_scalar_volume_field(ctx.models["neon_runtime"], "T").boundary_data_value()
    )
    return {
        "T": _host(ctx.fields["T"].internal_vector()),
        "time": {"value": ctx.time.value, "index": ctx.time.index, "deltaT": ctx.time.delta_t},
        "patches": {
            patch.patch: {
                "type": type(patch).__name__,
                "points": np.asarray(patch.points()).tolist(),
                "wall_T": np.asarray(patch.wall_T()).tolist(),
                "face_T": _host(ctx.fields["T"].boundary_data_value())[patch.faces],
                "written_T": written[patch.faces],
            }
            for patch in ctx.models["conduction"].patches
        },
    }


def _passes(ctx: Context) -> list[int]:
    operations = Coupling.of(ctx).operations()
    op = operations[0].func if len(operations) else None
    if isinstance(op, CouplingOp) and op.controller is not None:
        return [outcome.passes for outcome in op.controller.outcomes]
    return []


def main(mode: str) -> dict[str, Any]:
    if mode == "single":
        return {"regions": {"case": _region(run_single(["heatConductionNeoN"]))}}
    regions = run_multi()
    passes = {name: _passes(ctx) for name, ctx in regions.items()}
    return {
        "regions": {name: _region(ctx) for name, ctx in regions.items()},
        "passes": {name: counts for name, counts in passes.items() if counts},
    }


if __name__ == "__main__":
    print(RESULT_PREFIX + json.dumps(main(sys.argv[1])), flush=True)
