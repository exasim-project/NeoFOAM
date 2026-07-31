# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Assemble the VoF momentum equation with and without the MRF zones.

The incompressibleVoF twin of ``test/solver/incompressibleFluid/_mrf_worker.py``:
one ``Foam::Time`` per process, three assemblies through the *production*
``momentum`` operation that differ only in the injected ``mrf_zones``, and the
third one repeated without the zones so the source shift isolates the frame
acceleration from the wall-velocity correction the second one applied.

Here the term is the mass-weighted ``MRF.DDt(rho, U)`` of ``interFoam``'s
``UEqn.H``, so the shift must carry the mixture density. The case sets
``momentumPredictor no``, so no call solves.

Usage: ``python _mrf_worker.py <case-dir>``; writes ``<case-dir>/mrf.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.pressure_velocity.pimpleAlgorithm import momentum


def _assemble(ctx: Any, mrf_zones: Any) -> list[list[float]]:
    """Run the production momentum operation; return its matrix source."""
    updates = momentum(
        U=ctx.fields["U"],
        rho=ctx.fields["rho"],
        rhoPhi=ctx.fields["rhoPhi"],
        p_rgh=ctx.fields["p_rgh"],
        gh=ctx.fields["gh"],
        ghf=ctx.fields["ghf"],
        pimple_control=ctx.models["pimple_control"],
        mixture=ctx.models["mixture"],
        turbulence=ctx.models["turbulence"],
        mrf_zones=mrf_zones,
    )
    return np.asarray(updates["UEqn"].source()).tolist()


if __name__ == "__main__":
    case_dir = Path(sys.argv[1]).resolve()
    os.chdir(case_dir)

    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()
    mrf_zones = ctx.models["mrf_zones"]

    source_plain = _assemble(ctx, None)
    source_mrf = _assemble(ctx, mrf_zones)
    source_plain_corrected_walls = _assemble(ctx, None)

    (case_dir / "mrf.json").write_text(
        json.dumps(
            {
                "zones": len(mrf_zones),
                "cell_volumes": np.asarray(ctx.mesh.V()).tolist(),
                "source_with_mrf": source_mrf,
                "source_without_mrf": source_plain,
                "source_without_mrf_corrected_walls": source_plain_corrected_walls,
            },
            indent=1,
        )
    )
