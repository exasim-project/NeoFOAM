# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN ``kEpsilon`` closure survives a step taken at near-zero ``k``.

``kEpsilon`` divides by ``k`` — the ``C2 epsilon/k`` dissipation coefficient and the
``epsilon/k`` sink of the k equation — while ``k`` is only floored at
``kMin = 1e-15`` after each solve. A step taken with ``k`` already many orders below
its ``epsilon`` therefore drives those ratios to ~1e9, which is the arithmetic that
precedes an ``FE_OVERFLOW`` (the ``FOAM_SIGFPE`` "Floating point exception" the
verification sweep hit) once a run starts to diverge.

The case (``near_zero_k_base``) is ``walled_base`` — a 4 x 4 x 4 box whose two z
patches are walls, with ``kqRWallFunction`` / ``epsilonWallFunction`` / ``nutk``
boundaries and a linear solver at ``tolerance 1e-14, relTol 0`` — with ``k`` seeded
at ``1e-8`` against the stock ``epsilon = 14.855``, i.e. ``epsilon/k ~ 1.5e9``.

Correctness, not mere survival, is the assertion: OpenFOAM's own ``kEpsilon`` takes
the same step from the same files, so the two must still agree. The tolerance is
loose relative to the parity test's ``1e-8`` because both linear solves stop at an
absolute residual of ``1e-14``, which is a large *relative* slack for fields of
order ``1e-9``.

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
_CASE = _HERE / "near_zero_k_base"

#: The closure's floors (``models.kEpsilon.kMin`` / ``epsilonMin``): a solved field
#: may sit on them, never below.
K_MIN = 1e-15
EPSILON_MIN = 1e-15

#: Relative agreement with pybFoam. Both solves stop at an absolute residual of
#: 1e-14 (``near_zero_k_base/system/fvSolution``), which for a ``k`` of order 1e-9
#: is a relative slack of ~1e-5; 1e-4 keeps that headroom without hiding a
#: coefficient- or term-level disagreement, which would be O(1) here.
RTOL = 1.0e-4


def _run_worker(role: str, case: Path) -> None:
    """Run one worker role in a fresh process (one ``Foam::Time`` per process)."""
    subprocess.run(
        [sys.executable, str(_WORKER), role, str(case)],
        check=True,
        cwd=str(case.parent),
    )


@pytest.mark.parametrize("field", ["k", "epsilon", "nut"])
def test_near_zero_k_step_matches_pybfoam_and_stays_bounded(field: str, tmp_path: Path) -> None:
    """One ``correct`` step at ``epsilon/k ~ 1e9`` stays finite, bounded and on parity."""
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)

    _run_worker("mesh", case)
    _run_worker("reference", case)  # OpenFOAM's kEpsilon, same files
    _run_worker("subject", case)  # the NeoN closure

    reference = np.load(case / f"reference_{field}.npy")
    result = np.load(case / f"subject_{field}.npy")

    assert np.isfinite(result).all(), f"near_zero_k: {field} is not finite after one step"
    assert result.min() >= min(K_MIN, EPSILON_MIN), (
        f"near_zero_k: {field} fell below the closure's floor (min {result.min():g})"
    )
    np.testing.assert_allclose(
        result,
        reference,
        rtol=RTOL,
        err_msg=f"near_zero_k: {field} differs from pybFoam after one correct step",
    )
