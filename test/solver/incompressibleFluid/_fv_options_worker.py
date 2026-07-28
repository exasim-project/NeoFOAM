# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run one momentum + continuity pass, with or without the case's ``fv::options``.

One ``Foam::Time`` per process, so the staged init runs here and the test reads
the JSON. Both operations are the *production* ones, the only difference being
the injected ``fv_options`` — which is exactly what the solver's dependency
resolver varies between a case that carries an ``fvOptions`` dictionary and any
other case.

The two variants run in two processes rather than twice in one, so each starts
from the pristine ``0/`` state: the momentum matrices are then assembled from the
same fields, the linear solves get the same initial guess, and a difference
between the two velocities is attributable to the fvOptions hooks alone.

Usage: ``python _fv_options_worker.py <case-dir> {with,without}``; writes
``<case-dir>/fv_options-<variant>.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.solver.incompressibleFluid.models.pressure_velocity import (
    pimpleAlgorithm,
    simpleAlgorithm,
)


def _algorithm(ctx: Any) -> Any:
    """The module whose operations are production for this case.

    Which one is production is the case's own choice — the ``SIMPLE`` /
    ``PIMPLE`` dict of ``system/fvSolution``, already resolved into the control
    object the staged init put on the Context.
    """
    return simpleAlgorithm if "simple_control" in ctx.models else pimpleAlgorithm


def _control(ctx: Any) -> dict[str, Any]:
    if "simple_control" in ctx.models:
        return {"simple_control": ctx.models["simple_control"]}
    return {"pimple_control": ctx.models["pimple_control"]}


if __name__ == "__main__":
    case_dir = Path(sys.argv[1]).resolve()
    variant = sys.argv[2]
    os.chdir(case_dir)

    ctx = incompressibleFluid.instantiate(argv=["incompressibleFluid"]).initialize()
    U = ctx.fields["U"]
    fv_options = ctx.models["fv_options"] if variant == "with" else None

    algorithm = _algorithm(ctx)
    updates = algorithm.momentum(
        U=U,
        phi=ctx.fields["phi"],
        p=ctx.fields["p"],
        viscousStress=ctx.models["viscousStress"],
        ctx=ctx,
        fv_options=fv_options,
        **_control(ctx),
    )
    source = np.asarray(updates["UEqn"].source()).tolist()
    U_after_momentum = np.asarray(U.internalField()).copy().tolist()

    algorithm.continuity(
        U=U,
        p=ctx.fields["p"],
        phi=ctx.fields["phi"],
        UEqn=updates["UEqn"],
        cumulativeContErr=ctx.models["cumulativeContErr"],
        pressure_reference=ctx.models["pressure_reference"],
        fv_options=fv_options,
        **_control(ctx),
    )
    U_after_continuity = np.asarray(U.internalField()).copy().tolist()

    (case_dir / f"fv_options-{variant}.json").write_text(
        json.dumps(
            {
                "options": len(ctx.models["fv_options"]),
                "cell_centres": np.asarray(ctx.mesh.C().internalField()).tolist(),
                "cell_volumes": np.asarray(ctx.mesh.V()).tolist(),
                "source": source,
                "U_after_momentum": U_after_momentum,
                "U_after_continuity": U_after_continuity,
            },
            indent=1,
        )
    )
