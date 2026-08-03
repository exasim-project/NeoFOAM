# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run one VoF momentum + continuity pass, with or without the case's ``fv::options``.

The incompressibleVoF twin of
``test/solver/incompressibleFluid/_fv_options_worker.py``: one ``Foam::Time`` per
process, both operations the production ones, the only difference being the
injected ``fv_options``. Here the source call is the mass-weighted
``fvOptions(rho, U)`` of ``interFoam``'s ``UEqn.H``.

Usage: ``python _fv_options_worker.py <case-dir> {with,without}``; writes
``<case-dir>/fv_options-<variant>.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.pressure_velocity.pimpleAlgorithm import (
    continuity,
    momentum,
)

if __name__ == "__main__":
    case_dir = Path(sys.argv[1]).resolve()
    variant = sys.argv[2]
    os.chdir(case_dir)

    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()
    U = ctx.fields["U"]
    fv_options = ctx.models["fv_options"] if variant == "with" else None

    updates = momentum(
        U=U,
        rho=ctx.fields["rho"],
        rhoPhi=ctx.fields["rhoPhi"],
        p_rgh=ctx.fields["p_rgh"],
        gh=ctx.fields["gh"],
        ghf=ctx.fields["ghf"],
        pimple_control=ctx.models["pimple_control"],
        mixture=ctx.models["mixture"],
        turbulence=ctx.models["turbulence"],
        fv_options=fv_options,
    )
    source = np.asarray(updates["UEqn"].source()).tolist()
    U_after_momentum = np.asarray(U.internalField()).copy().tolist()

    continuity(
        U=U,
        rho=ctx.fields["rho"],
        p_rgh=ctx.fields["p_rgh"],
        p=ctx.fields["p"],
        phi=ctx.fields["phi"],
        gh=ctx.fields["gh"],
        ghf=ctx.fields["ghf"],
        pimple_control=ctx.models["pimple_control"],
        cumulativeContErr=ctx.models["cumulativeContErr"],
        pressure_reference=ctx.models["pressure_reference"],
        mixture=ctx.models["mixture"],
        last_rAU=ctx.models["last_rAU"],
        Uf=ctx.models.get("Uf"),
        fv_options=fv_options,
        UEqn=updates["UEqn"],
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
