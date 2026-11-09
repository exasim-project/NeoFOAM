# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the coupled ``multiRun`` case in the working directory; print what it computed as JSON.

The fluid (incompressibleFluid) and the plate (heatConductionNeoN) of
``cases/heatedPlate``. The answer is the single stdout line prefixed
``#RESULT``: the fluid's coupled patches (condition, face centres, wall
temperature, heat flux into the fluid), the plate's wall temperature and heat
flux out of it on the same interface, both times, and the passes of the
leader's coupling per time step.

Every message of the multiRun comm_channel is pickled, as between processes
(``InProcessCommChannel.PICKLE``). A subprocess: Kokkos and ``Foam::Time`` are
per-process, and NeoN must be finalized after every NeoN object is gone, so the
contexts never outlive :func:`main`.
"""

import json
from typing import Any

import numpy as np

from neofoam.framework.coupling import Coupling, InProcessCommChannel
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.solver.multiRun import run as run_multi

RESULT_PREFIX = "#RESULT "


def main() -> dict[str, Any]:
    InProcessCommChannel.PICKLE = True
    regions = run_multi()
    fluid, solid = regions["fluid"], regions["solid"]
    passes: dict[str, list[int]] = {}
    for name, ctx in regions.items():
        operations = Coupling.of(ctx).operations()
        op = operations[0].func if len(operations) else None
        if isinstance(op, CouplingOp) and op.controller is not None:
            passes[name] = [outcome.passes for outcome in op.controller.outcomes]
    (top,) = solid.models["conduction"].patches
    return {
        "fluid": {
            patch.patch: {
                "type": type(patch).__name__,
                "condition": patch.condition,
                "points": np.asarray(patch.points()).tolist(),
                "areas": np.asarray(fluid.fields["T"].mesh().magSf()[patch.patch]).tolist(),
                "wall_T": np.asarray(patch.wall_T()).tolist(),
                "q": np.asarray(patch.heat_flux()).tolist(),
            }
            for patch in fluid.models["coupled_patches"]
        },
        "solid": {
            "wall_T": np.asarray(top.wall_T()).tolist(),
            "q_out": (-np.asarray(top.heat_in)).tolist(),
            "areas": np.asarray(top.areas).tolist(),
        },
        "time": {name: ctx.time.value for name, ctx in regions.items()},
        "passes": passes,
    }


if __name__ == "__main__":
    print(RESULT_PREFIX + json.dumps(main()), flush=True)
