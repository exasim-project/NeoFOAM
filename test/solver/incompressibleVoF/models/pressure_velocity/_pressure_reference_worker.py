# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run the VoF staged init once and exercise the pressure-reference helpers.

Run as ``python _pressure_reference_worker.py <case_dir>``; writes
``<case_dir>/pressure_reference.json``. A process owns exactly one
``Foam::Time``, so the closed and the open case each need their own process —
this worker is the single pass one of them gets.

The pipeline is driven exactly as production does it
(``create_init(case_dir)`` -> ``runner.argv`` -> ``runner.run()``), then
``update_absolute_pressure`` is applied once — the same call the ``continuity``
operation makes at the tail of every pressure corrector — and the fields are
dumped before and after it, so the test can pin both the reconstruction
``p = p_rgh + rho*gh`` and the closed-domain level shift.

``*_before`` is therefore the state the *init* leaves behind, which on a closed
domain is already levelled: the ``pressure_reference`` step runs createFields.H's
own ``if (p_rgh.needReference())`` block.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
    runner.argv = ["incompressibleVoF"]
    ctx = runner.run()

    p = ctx.fields["p"]
    p_rgh = ctx.fields["p_rgh"]
    rho = ctx.fields["rho"]
    gh = ctx.fields["gh"]
    reference = ctx.models["pressure_reference"]

    result: dict[str, Any] = {
        "pressure_reference": reference,
        # need_reference() called straight on setRefCell's cell index,
        # independently of the flag the init step stored, so the test can pin
        # the helper itself.
        "need_reference": need_reference(reference["pRefCell"]),
        "rho": _internal(rho),
        "gh": _internal(gh),
        "p_rgh_before": _internal(p_rgh),
        "p_before": _internal(p),
        # getRefCellValue on every cell of a known field (hand-derivable) plus
        # the "no rank owns a reference cell" case, which OpenFOAM's
        # returnReduce(0, sumOp) answers with 0.
        "ref_cell_values_of_p_rgh": [
            get_ref_cell_value(p_rgh, cell) for cell in range(ctx.mesh.nCells())
        ],
        "ref_cell_value_without_reference_cell": get_ref_cell_value(p_rgh, -1),
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
    (case_dir / "pressure_reference.json").write_text(json.dumps(run(case_dir), indent=1))


if __name__ == "__main__":
    main()
