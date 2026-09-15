# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Record the ``Foam::Time`` state every alpha sub-cycle pass sees.

Run as ``python _sub_cycle_worker.py <case_dir>``; writes
``<case_dir>/sub_cycle.json``. A process owns exactly one ``Foam::Time``, hence
a worker: the whole point of the dump is what ``Time`` does *inside* one
``alpha_advection`` call, which needs a real initialised case.

``mules.alpha_eqn`` is wrapped with a recorder before the call — it is the one
place per sub-step where the time-dependent boundary conditions are evaluated,
so what it sees is exactly what a ``waveAlpha``/``waveVelocity`` patch would.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.alpha_advection.models import mules


def run(case_dir: Path) -> dict[str, Any]:
    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()

    runtime = ctx.models["runtime"]
    runtime.increment()  # one real time step, as the solver's time loop does

    seen: list[dict[str, Any]] = []
    real_alpha_eqn = mules.alpha_eqn

    def recording_alpha_eqn(*args: Any, **kwargs: Any) -> Any:
        seen.append(
            {
                "time": runtime.value(),
                "deltaT": runtime.deltaTValue(),
                "timeIndex": runtime.timeIndex(),
            }
        )
        return real_alpha_eqn(*args, **kwargs)

    before = {
        "time": runtime.value(),
        "deltaT": runtime.deltaTValue(),
        "timeIndex": runtime.timeIndex(),
    }
    mules.alpha_eqn = recording_alpha_eqn
    try:
        mules.alpha_advection(
            ctx.fields["alpha1"],
            ctx.fields["alpha2"],
            ctx.fields["phi"],
            ctx.fields["rhoPhi"],
            ctx.fields["rho"],
            ctx.fields["alphaPhiUn"],
            ctx.fields["alphaPhi10"],
            ctx.models["mixture"],
            ctx.models["alphaPhi1Corr0"],
        )
    finally:
        mules.alpha_eqn = real_alpha_eqn

    return {
        "before": before,
        "sub_steps": seen,
        "after": {
            "time": runtime.value(),
            "deltaT": runtime.deltaTValue(),
            "timeIndex": runtime.timeIndex(),
        },
    }


if __name__ == "__main__":
    case = Path(sys.argv[1]).resolve()
    os.chdir(case)
    (case / "sub_cycle.json").write_text(json.dumps(run(case), indent=1))
