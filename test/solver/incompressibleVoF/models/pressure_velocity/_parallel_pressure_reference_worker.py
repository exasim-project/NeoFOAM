# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Per-rank dump of the pressure-reference helpers under ``-parallel``.

Launched once per MPI rank as ``mpirun -np N python
_parallel_pressure_reference_worker.py <case-dir>`` from a decomposed case;
each rank writes ``<case-dir>/parallel_pressure_reference_<rank>.json``. The
serial sibling ``_pressure_reference_worker.py`` does exactly the same thing on
one rank — this one keeps the pipeline identical (``create_init`` ->
``runner.argv`` -> ``runner.run()``, then one ``update_absolute_pressure``) and
only adds ``-parallel`` and the rank identity, so the two dumps can be compared
field by field.

The raw ``pRefCell`` is dumped *next to* ``need_reference``: OpenFOAM's
``setRefCell`` hands a non-negative index only to the rank that owns the
reference cell, so the two disagreeing is the whole point of the reduction the
test is there to check.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybFoam as pyf

from neofoam.solver.incompressibleVoF.create_fields import create_init
from neofoam.solver.incompressibleVoF.models.pressure_velocity.pressure_reference import (
    get_ref_cell_value,
    need_reference,
    update_absolute_pressure,
)


def _internal(field: Any) -> Any:
    """Internal field values as plain Python lists."""
    return np.asarray(field.internalField()).tolist()


def run(case_dir: Path) -> dict[str, Any]:
    runner = create_init(case_dir=case_dir)
    runner.argv = ["incompressibleVoF", "-parallel"]
    ctx = runner.run()

    p = ctx.fields["p"]
    p_rgh = ctx.fields["p_rgh"]
    rho = ctx.fields["rho"]
    gh = ctx.fields["gh"]
    reference = ctx.models["pressure_reference"]

    result: dict[str, Any] = {
        "rank": pyf.Pstream.myProcNo(),
        "nProcs": pyf.Pstream.nProcs(),
        "parRun": pyf.Pstream.parRun(),
        # Cell centres, so the test can put the per-rank slices back in global
        # order without assuming how decomposePar numbered them.
        "cell_centres_x": np.asarray(ctx.mesh.C().internalField())[:, 0].tolist(),
        "pressure_reference": reference,
        # ``need_reference`` called straight on setRefCell's own cell index, so
        # the test pins the helper and not the flag the init step cached.
        "need_reference": need_reference(reference["pRefCell"]),
        "ref_cell_value_of_p_rgh": get_ref_cell_value(p_rgh, reference["pRefCell"]),
        "ref_cell_value_without_reference_cell": get_ref_cell_value(p_rgh, -1),
        "rho": _internal(rho),
        "gh": _internal(gh),
        "p_before": _internal(p),
    }

    update_absolute_pressure(
        p,
        p_rgh,
        rho,
        gh,
        ref_cell=reference["pRefCell"],
        ref_value=reference["pRefValue"],
        needs_reference=reference["needsRef"],
    )

    result["p_after"] = _internal(p)
    result["p_rgh_after"] = _internal(p_rgh)
    result["ref_cell_value_of_p_after"] = get_ref_cell_value(p, reference["pRefCell"])
    return result


def main() -> None:
    case_dir = Path(sys.argv[1]).resolve()
    os.chdir(case_dir)
    result = run(case_dir)
    (case_dir / f"parallel_pressure_reference_{result['rank']}.json").write_text(
        json.dumps(result, indent=1)
    )


if __name__ == "__main__":
    main()
