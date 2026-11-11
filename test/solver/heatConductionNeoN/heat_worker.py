# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run a heatConductionNeoN case (``single|multi|geometry|fields``); print ``#RESULT`` JSON.

A subprocess: Kokkos and ``Foam::Time`` are per process, and NeoN must be finalized
after its objects are gone.
"""

import json
import sys
from typing import Any

import numpy as np
from framework.coupling.regions import Outcomes

from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import Context
from neofoam.framework.coupling import Cells, Patch, coupling
from neofoam.framework.coupling.loop._loop_op import CouplingLoopOp
from neofoam.framework.execution_observer import observe
from neofoam.solver.heatConductionNeoN import heatConductionNeoN
from neofoam.solver.heatConductionNeoN import run as run_single
from neofoam.solver.multiRun import run as run_multi
from neofoam.solver.neon_runtime import NeoNFields, NeoNGeometry

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
                "points": np.asarray(coupling(ctx).geometry.face_centres(patch.patch)).tolist(),
                "wall_T": np.asarray(patch.wall_T()).tolist(),
                "face_T": _host(ctx.fields["T"].boundary_data_value())[patch.faces],
                "written_T": written[patch.faces],
            }
            for patch in ctx.models["conduction"].patches
        },
    }


def _passes(ctx: Context, outcomes: Outcomes) -> list[int]:
    operations = coupling(ctx).operations()
    op = operations[0].func if len(operations) else None
    if isinstance(op, CouplingLoopOp) and op.controller is not None:
        return [outcome.passes for outcome in outcomes.of(op.controller.label)]
    return []


def _geometry() -> dict[str, Any]:
    """NeoN's geometry of the case beside OpenFOAM's, per patch and for the cells."""
    ctx = heatConductionNeoN.instantiate(argv=["heatConductionNeoN"]).initialize()
    runtime = ctx.models["neon_runtime"]
    mesh, neon = runtime.mesh, NeoNGeometry(runtime)
    boundary = mesh.boundary()
    patches = [str(boundary[index].name()) for index in range(len(boundary))]
    return {
        "patches": {
            name: {
                "neon": {
                    "centres": neon.face_centres(name).tolist(),
                    "areas": neon.face_areas(name).tolist(),
                },
                "foam": {
                    "centres": np.asarray(mesh.Cf()[name]).tolist(),
                    "areas": np.asarray(mesh.magSf()[name]).tolist(),
                },
            }
            for name in patches
        },
        "cells": {
            "neon": {
                "centres": neon.cell_centres().tolist(),
                "volumes": neon.cell_volumes().tolist(),
            },
            "foam": {
                "centres": np.asarray(mesh.C().internalField()).tolist(),
                "volumes": np.asarray(mesh.V()).tolist(),
            },
        },
    }


def _fields() -> dict[str, Any]:
    """What NeoNFields reads of T, before and after it writes a patch and the cells."""
    ctx = heatConductionNeoN.instantiate(argv=["heatConductionNeoN"]).initialize()
    fields = NeoNFields(ctx.models["neon_runtime"], ctx.fields)
    read = {
        "outer": fields.read("T", Patch("outer")).tolist(),
        "cells": fields.read("T", Cells()).tolist(),
    }
    fields.write("T", Patch("interface"), [301.0, 302.0, 303.0, 304.0, 305.0])
    fields.write("T", Cells(), np.arange(30.0))
    written = {
        "interface": fields.read("T", Patch("interface")).tolist(),
        "cells": fields.read("T", Cells()).tolist(),
    }
    return {"read": read, "written": written}


def main(mode: str) -> dict[str, Any]:
    if mode == "geometry":
        return _geometry()
    if mode == "fields":
        return _fields()
    if mode == "single":
        return {"regions": {"case": _region(run_single(["heatConductionNeoN"]))}}
    outcomes = Outcomes()
    with observe(outcomes):
        regions = run_multi()
    passes = {name: _passes(ctx, outcomes) for name, ctx in regions.items()}
    return {
        "regions": {name: _region(ctx) for name, ctx in regions.items()},
        "passes": {name: counts for name, counts in passes.items() if counts},
    }


if __name__ == "__main__":
    print(RESULT_PREFIX + json.dumps(main(sys.argv[1])), flush=True)
