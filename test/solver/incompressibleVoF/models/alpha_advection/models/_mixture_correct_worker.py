# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Record every ``mixture.correct()`` one alpha advection makes, and when.

Run as ``python _mixture_correct_worker.py <case_dir>``; writes
``<case_dir>/mixture_correct.json``. A process owns exactly one ``Foam::Time``,
hence a worker: the dump is about what happens *inside* one live
``alpha_advection`` call on a real initialised case.

The context's own mixture is handed to ``alpha_advection`` behind a delegating
proxy that logs ``Foam::Time``'s current ``deltaT`` on every ``correct()``. That
one number says which side of the sub-cycle the call is on: inside, ``Time`` is
sub-cycled and reads ``deltaT/nAlphaSubCycles``; after ``endSubCycle()`` it is
back to the real time step.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.alpha_advection.models import mules


class _CorrectRecorder:
    """The real mixture, with every ``correct()`` call timestamped by deltaT."""

    def __init__(self, mixture: Any, runtime: Any) -> None:
        self._mixture = mixture
        self._runtime = runtime
        self.delta_t_at_each_correct: list[float] = []

    def correct(self) -> None:
        self.delta_t_at_each_correct.append(self._runtime.deltaTValue())
        self._mixture.correct()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mixture, name)


def run(case_dir: Path) -> dict[str, Any]:
    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()

    runtime = ctx.models["runtime"]
    runtime.increment()  # one real time step, as the solver's time loop does

    mixture = _CorrectRecorder(ctx.models["mixture"], runtime)
    mules.alpha_advection(
        ctx.fields["alpha1"],
        ctx.fields["alpha2"],
        ctx.fields["phi"],
        ctx.fields["rhoPhi"],
        ctx.fields["rho"],
        ctx.fields["alphaPhiUn"],
        ctx.fields["alphaPhi10"],
        mixture,
        ctx.models["alphaPhi1Corr0"],
    )

    return {"delta_t_at_each_correct": mixture.delta_t_at_each_correct}


if __name__ == "__main__":
    case = Path(sys.argv[1]).resolve()
    os.chdir(case)
    (case / "mixture_correct.json").write_text(json.dumps(run(case), indent=1))
