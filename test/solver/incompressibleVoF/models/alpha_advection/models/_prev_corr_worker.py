# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Record the ``talphaPhi1Corr0`` compression-flux cache over three time steps.

Run as ``python _prev_corr_worker.py <case_dir>``; writes
``<case_dir>/prev_corr.json``. A process owns exactly one ``Foam::Time``, and
the whole point of the cache is that it survives *between* alpha passes, so it
takes a real initialised case driven over three time steps rather than the
one-shot ``vof_row4`` fixture. Three, not two: this case's first pass is
Courant-limited to a zero correction, so a cached correction only becomes
non-zero — and only starts changing anything — from the third.

``mules_implicit_predictor`` is wrapped before the call so each pass's pure
upwind flux is captured as well: it is the base ``alphaEqn.H:230`` measures the
cached correction against, and the only way to check that identity from outside.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.alpha_advection.models import mules

_N_TIME_STEPS = 3


def _internal(field: Any) -> list[float]:
    return list(np.asarray(field.internalField()).tolist())


def _cache(slot: list[Optional[Any]]) -> Optional[list[float]]:
    return None if slot[0] is None else _internal(slot[0])


def run(case_dir: Path) -> dict[str, Any]:
    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()

    runtime = ctx.models["runtime"]
    slot = ctx.models["alphaPhi1Corr0"]

    upwind: list[list[float]] = []
    real_predictor = mules.mules_implicit_predictor

    def recording_predictor(*args: Any, **kwargs: Any) -> Any:
        flux = real_predictor(*args, **kwargs)
        upwind.append(_internal(flux))
        return flux

    mules.mules_implicit_predictor = recording_predictor
    passes: list[dict[str, Any]] = []
    try:
        for _ in range(_N_TIME_STEPS):
            runtime.increment()
            cache_before = _cache(slot)
            mules.alpha_advection(
                ctx.fields["alpha1"],
                ctx.fields["alpha2"],
                ctx.fields["phi"],
                ctx.fields["rhoPhi"],
                ctx.fields["rho"],
                ctx.fields["alphaPhiUn"],
                ctx.fields["alphaPhi10"],
                ctx.models["mixture"],
                slot,
            )
            passes.append(
                {
                    "cache_before": cache_before,
                    "cache_after": _cache(slot),
                    "alphaPhi10": _internal(ctx.fields["alphaPhi10"]),
                    "upwind": upwind[-1],
                }
            )
    finally:
        mules.mules_implicit_predictor = real_predictor

    return {"passes": passes}


if __name__ == "__main__":
    case = Path(sys.argv[1]).resolve()
    os.chdir(case)
    (case / "prev_corr.json").write_text(json.dumps(run(case), indent=1))
