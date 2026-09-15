# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Record the alpha off-centring a Crank-Nicolson case produces, step by step.

Run as ``python _crank_nicolson_worker.py <case_dir>``; writes
``<case_dir>/crank_nicolson.json``. A process owns exactly one ``Foam::Time``,
hence a worker: the off-centring depends on how far the run has advanced, so it
takes a real initialised case driven over two time steps.

``alpha_advection`` is called on each step so the run is the one the solver
would do; a rejected scheme combination surfaces as ``error`` instead of a
second ``oc_coeff`` entry.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.alpha_advection.models.mules import (
    alpha_advection,
    read_alpha_controls,
    read_alpha_ddt_off_centring,
)

_N_TIME_STEPS = 2


def run(case_dir: Path) -> dict[str, Any]:
    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()

    runtime = ctx.models["runtime"]
    alpha1 = ctx.fields["alpha1"]
    _, n_alpha_sub_cycles, _, _ = read_alpha_controls(alpha1.name())

    oc_coeffs: list[float] = []
    for _ in range(_N_TIME_STEPS):
        runtime.increment()
        try:
            oc_coeffs.append(read_alpha_ddt_off_centring(alpha1.mesh(), n_alpha_sub_cycles))
            alpha_advection(
                alpha1,
                ctx.fields["alpha2"],
                ctx.fields["phi"],
                ctx.fields["rhoPhi"],
                ctx.fields["rho"],
                ctx.fields["alphaPhiUn"],
                ctx.fields["alphaPhi10"],
                ctx.models["mixture"],
                ctx.models["alphaPhi1Corr0"],
            )
        except ValueError as err:
            return {"oc_coeffs": oc_coeffs, "error": str(err)}

    return {"oc_coeffs": oc_coeffs, "error": None}


if __name__ == "__main__":
    case = Path(sys.argv[1]).resolve()
    os.chdir(case)
    (case / "crank_nicolson.json").write_text(json.dumps(run(case), indent=1))
