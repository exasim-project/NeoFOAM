# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity of the NeoN closures' **dilatation terms** against pybFoam.

OpenFOAM's ``kEpsilon``/``kOmegaSST`` carry ``fvm::SuSp(divU, ...)`` dilatation
terms (``kEpsilon.C``, ``kOmegaSSTBase.C``) with ``divU = fvc::div(phi)`` — zero
only once the pressure solve has converged. Every other parity case here seeds a
divergence-free velocity on purpose (see :mod:`test_wall_function_parity`), so a
closure that drops these terms still matches to round-off there while drifting
~1e-3 of peak per ``correct`` step on a real SIMPLE iterate, where the continuity
defect is O(1) — the pitzDaily frozen-state attribution that motivated this test.

``dilatation_base`` is ``wall_function_base`` with ``U_y += (y - 0.5)^2``: the
flux keeps zero wall-normal component (the wall functions stay exercised
unchanged) but ``div U = 2(y - 0.5)`` changes sign across the box, so the
implicit (positive-coefficient) and explicit (negative-coefficient) halves of
OpenFOAM's ``SuSp`` split each bind in half of the cells within one run.

Bounds are the wall-function module's, measured against the same 1e-14
``fvSolution`` tolerances: round-off for ``kEpsilon``; 1e-6 for ``kOmegaSST``,
whose ``F1``/``F2`` blending chain associates differently on the two backends.
One ``Foam::Time`` per process, so the roles run as subprocesses of
:mod:`_parity_worker` and hand their fields over as ``.npy`` — see that module.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
_WORKER = _HERE / "_parity_worker.py"
_CASE = _HERE / "dilatation_base"
_MODELS = _HERE / "parity_models"

#: ``(model, fields it owns, agreement bound as a fraction of the field's peak)``.
CASES = [
    ("kEpsilon", ("nut", "k", "epsilon"), 1e-12),
    ("kOmegaSST", ("nut", "k", "omega"), 1e-6),
]


def _run_worker(role: str, case: Path) -> None:
    """Run one worker role in a fresh process (one ``Foam::Time`` per process)."""
    subprocess.run(
        [sys.executable, str(_WORKER), role, str(case)],
        check=True,
        cwd=str(case.parent),
    )


@pytest.mark.parametrize(("model", "fields", "bound"), CASES, ids=[c[0] for c in CASES])
def test_fields_match_pybfoam_with_nonconservative_flux(
    model: str, fields: tuple[str, ...], bound: float, tmp_path: Path
) -> None:
    """One ``correct`` step on a non-conservative flux matches pybFoam."""
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)
    shutil.copyfile(
        _MODELS / model / "turbulenceProperties",
        case / "constant" / "turbulenceProperties",
    )

    _run_worker("mesh", case)  # the case ships its own 0/ fields; do not reseed
    _run_worker("reference", case)  # pybFoam: dilatation terms via fvm::SuSp
    _run_worker("subject", case)  # the NeoN closure's dilatation terms

    for name in fields:
        reference = np.load(case / f"reference_{name}.npy")
        result = np.load(case / f"subject_{name}.npy")
        peak = float(np.max(np.abs(reference))) or 1.0
        np.testing.assert_allclose(
            result,
            reference,
            rtol=0.0,
            atol=bound * peak,
            err_msg=f"{model}: {name} differs after one correct step on dilatation_base",
        )
