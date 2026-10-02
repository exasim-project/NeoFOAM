# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the VoF staged init once and exercise one MULES alpha solve.

Run as ``python _alpha_phi_un_worker.py <case_dir>``; writes
``<case_dir>/alpha_phi_un.json``. A process owns exactly one ``Foam::Time``, so
this gets its own worker rather than sharing ``conftest.py``'s ``vof_row4``
fixture: BUILD only zero-initialises and registers ``alphaPhiUn``
(``shared_field_build_steps``) — it never solves — so the fixture alone cannot
show the "updated after an alpha solve" half of the spec.

The pipeline is driven exactly as production does it (``create_init(case_dir)``
-> ``runner.argv`` -> ``runner.run()``), then the ``alpha_advection`` operation
itself is called directly with the context's own fields/model (the
``@mules.operation``/``@...FvSchemes.add``/``@...FvSolution.add`` decorators
are identity decorators — ``alpha_advection`` is still the plain function) —
and ``alphaPhiUn`` is dumped before and after.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.alpha_advection.models.mules import (
    alpha_advection,
)


def _internal(field: Any) -> Any:
    return np.asarray(field.internalField()).tolist()


def run(case_dir: Path) -> dict[str, Any]:
    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()

    alpha_phi_un = ctx.fields["alphaPhiUn"]
    result: dict[str, Any] = {
        "registered_name": alpha_phi_un.name(),
        "before": _internal(alpha_phi_un),
    }

    alpha_advection(
        ctx.fields["alpha1"],
        ctx.fields["alpha2"],
        ctx.fields["phi"],
        ctx.fields["rhoPhi"],
        ctx.fields["rho"],
        alpha_phi_un,
        ctx.fields["alphaPhi10"],
        ctx.models["mixture"],
        ctx.models["alphaPhi1Corr0"],
    )

    result["after"] = _internal(alpha_phi_un)
    # Still the very same registered object — ``.assign`` in place, not a
    # fresh same-named field that would have deregistered the original.
    result["still_registered_name"] = ctx.fields["alphaPhiUn"].name()
    return result


def main() -> None:
    case_dir = Path(sys.argv[1]).resolve()
    os.chdir(case_dir)
    (case_dir / "alpha_phi_un.json").write_text(json.dumps(run(case_dir), indent=1))


if __name__ == "__main__":
    main()
