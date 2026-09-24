# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Assemble the case's momentum equation with and without the MRF zones.

One ``Foam::Time`` per process, so the staged init runs here and the test reads
the JSON. Every matrix comes out of the *production* ``momentum`` operation, the
only difference being the injected momentum extensions — the ones the
case's own models resolve to, against the empty seam a case without
``MRFProperties`` would get.

Three assemblies, in this order, so both hooks can be read off independently:

1. without the zones — the boundary is still the ``0/U`` no-slip state;
2. with the zones — this is the call that must correct the rotating wall faces
   *and* add the frame acceleration;
3. without the zones again — same corrected boundary state as (2), so
   ``source(2) - source(3)`` isolates the frame acceleration alone.

The case sets ``momentumPredictor no``, so no call solves and ``U``'s internal
field still carries its ``0/U`` values throughout.

Usage: ``python _mrf_worker.py <case-dir>``; writes ``<case-dir>/mrf.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from neofoam.framework.model import BoundExtension
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.solver.incompressibleFluid.models.pressure_velocity.extension import (
    momentum_extension,
)
from neofoam.solver.incompressibleFluid.models.pressure_velocity.pimpleAlgorithm import (
    momentum as pimple_momentum,
)
from neofoam.solver.incompressibleFluid.models.pressure_velocity.simpleAlgorithm import (
    momentum as simple_momentum,
)


def _assemble(ctx: Any, ext: Any) -> list[list[float]]:
    """Run the production momentum operation; return its matrix source.

    Which one is production for this case is the case's own choice — the
    ``SIMPLE`` / ``PIMPLE`` dict of ``system/fvSolution``, already resolved into
    the control object the staged init put on the Context.
    """
    common = dict(
        U=ctx.fields["U"],
        phi=ctx.fields["phi"],
        p=ctx.fields["p"],
        viscousStress=ctx.models["viscousStress"],
        ctx=ctx,
        ext=ext,
    )
    if "simple_control" in ctx.models:
        updates = simple_momentum(simple_control=ctx.models["simple_control"], **common)
    else:
        updates = pimple_momentum(pimple_control=ctx.models["pimple_control"], **common)
    return np.asarray(updates["UEqn"].source()).tolist()


if __name__ == "__main__":
    case_dir = Path(sys.argv[1]).resolve()
    os.chdir(case_dir)

    ctx = incompressibleFluid.instantiate(argv=["incompressibleFluid"]).initialize()
    mesh = ctx.mesh
    U = ctx.fields["U"]
    mrf_zones = ctx.models["mrf_zones"]

    mrf_ext = momentum_extension.resolve(ctx)
    # Resolve against no Context: the seam of a case without the model — every
    # site falls back to its declared default (``+ ext.terms(U)`` adds the zero
    # seed only).
    no_ext: BoundExtension = momentum_extension.resolve(None)

    walls_before = np.asarray(U["walls"]).tolist()
    source_plain = _assemble(ctx, no_ext)
    source_mrf = _assemble(ctx, mrf_ext)
    walls_after = np.asarray(U["walls"]).tolist()
    source_plain_corrected_walls = _assemble(ctx, no_ext)

    (case_dir / "mrf.json").write_text(
        json.dumps(
            {
                "zones": len(mrf_zones),
                "cell_centres": np.asarray(mesh.C().internalField()).tolist(),
                "cell_volumes": np.asarray(mesh.V()).tolist(),
                "walls_face_centres": np.asarray(mesh.Cf()["walls"]).tolist(),
                "walls_U_before": walls_before,
                "walls_U_after": walls_after,
                "source_without_mrf": source_plain,
                "source_with_mrf": source_mrf,
                "source_without_mrf_corrected_walls": source_plain_corrected_walls,
            },
            indent=1,
        )
    )
