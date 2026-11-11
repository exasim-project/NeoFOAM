# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the coupled ``multiRun`` case in the working directory; print what it computed as JSON.

The answer is the stdout line prefixed ``#RESULT``. A subprocess, since Kokkos
and ``Foam::Time`` are per process and NeoN must be finalized after its objects
are gone, so no context outlives :func:`main`.
"""

import json
from typing import Any

import numpy as np
from framework.coupling.regions import Outcomes

from neofoam.framework.context import Context
from neofoam.framework.coupling import coupling
from neofoam.framework.coupling.loop._loop_op import CouplingLoopOp
from neofoam.framework.execution_observer import observe
from neofoam.solver.multiRun import run as run_multi

RESULT_PREFIX = "#RESULT "


def _passes(ctx: Context, outcomes: Outcomes) -> list[int]:
    operations = coupling(ctx).operations()
    op = operations[0].func if len(operations) else None
    if isinstance(op, CouplingLoopOp) and op.controller is not None:
        return [outcome.passes for outcome in outcomes.of(op.controller.label)]
    return []


def _fluid(ctx: Context) -> dict[str, Any]:
    mesh = ctx.fields["T"].mesh()
    return {
        patch.patch: {
            "type": type(patch).__name__,
            "points": np.asarray(coupling(ctx).geometry.face_centres(patch.patch)).tolist(),
            "areas": np.asarray(mesh.magSf()[patch.patch]).tolist(),
            "wall_T": np.asarray(patch.wall_T()).tolist(),
            "q": np.asarray(patch.heat_flux()).tolist(),
        }
        for patch in ctx.models["coupled_patches"]
    }


def main() -> dict[str, Any]:
    outcomes = Outcomes()
    with observe(outcomes):
        regions = run_multi()
    fluid, solid = regions["fluid"], regions["solid"]
    passes = {name: _passes(ctx, outcomes) for name, ctx in regions.items()}
    (top,) = solid.models["conduction"].patches
    return {
        "fluid": _fluid(fluid),
        "kept_entries": fluid.models["writer"].hook.entries,
        "solid": {
            "wall_T": np.asarray(top.wall_T()).tolist(),
            "q_out": (-np.asarray(top.heat_in)).tolist(),
            "areas": np.asarray(top.areas).tolist(),
        },
        "time": {name: ctx.time.value for name, ctx in regions.items()},
        "passes": {name: counts for name, counts in passes.items() if counts},
    }


if __name__ == "__main__":
    print(RESULT_PREFIX + json.dumps(main()), flush=True)
