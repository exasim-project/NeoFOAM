# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Sub-floor ``k`` / ``epsilon`` cells are repaired the way ``Foam::bound`` repairs them.

``Foam::bound`` (``src/finiteVolume/cfdTools/general/bound/bound.C``) does **not**
clip a non-positive cell to the floor. It gives it
``fvc::average(max(vsf, lower))`` — the face-area-weighted average of the floored
neighbourhood — and only then takes the max against the floor. The distinction is
not cosmetic for a ``kEpsilon`` closure: clipping ``epsilon`` to ``1e-15`` leaves
``nut = Cmu k^2 / epsilon`` dividing by ``1e-15``, which is how a single bad cell
turns into ``nut ~ 1e10`` and takes the run with it.

The case (``sub_floor_k_base``) is ``near_zero_k_base`` with four cells seeded
negative — two in ``k`` (both interior) and two in ``epsilon`` (one a ``zMin`` wall
cell, one interior), so the ``fvc::average`` stencil is exercised both with and
without a boundary face. OpenFOAM's own ``kEpsilon`` bounds them as it constructs
(``bound(k_, kMin_); bound(epsilon_, epsilonMin_)``) and so does the NeoN closure's
``build``; both then take the same ``correct`` step from the repaired fields, so
the comparison covers the repair *and* everything downstream of it.

Discriminating power: swapping the repair for the old hard floor
(``field_max(k, kMin)``) moves ``k`` and ``epsilon`` by ~100% of their own value in
the seeded cells and ``nut`` by a factor of ~1e4 — four orders outside
:data:`RTOL`, which is set from the measured agreement (~1.5e-10).

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
_CASE = _HERE / "sub_floor_k_base"

#: The closure's floors (``models.kEpsilon.kMin`` / ``epsilonMin``). A *repaired*
#: cell must land well above them — landing *on* one is the bug this test exists for.
K_MIN = 1e-15
EPSILON_MIN = 1e-15

#: Relative agreement with pybFoam. Measured 1.5e-10 across k / epsilon / nut, set
#: by the 1e-14 absolute linear-solver tolerance both backends stop at
#: (``sub_floor_k_base/system/fvSolution``) against fields of order 1e-9 .. 1e-4.
#: 1e-8 keeps that headroom and still leaves four orders of margin to the ~100%
#: error the hard-floor repair produces.
RTOL = 1.0e-8


def _run_worker(role: str, case: Path) -> None:
    """Run one worker role in a fresh process (one ``Foam::Time`` per process)."""
    subprocess.run(
        [sys.executable, str(_WORKER), role, str(case)],
        check=True,
        cwd=str(case.parent),
    )


@pytest.mark.parametrize("field", ["k", "epsilon", "nut"])
def test_sub_floor_cells_are_bounded_like_openfoam(field: str, tmp_path: Path) -> None:
    """A ``correct`` step from four sub-floor cells matches pybFoam cell-by-cell."""
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)

    _run_worker("mesh", case)
    _run_worker("reference", case)  # OpenFOAM's kEpsilon, same files
    _run_worker("subject", case)  # the NeoN closure

    reference = np.load(case / f"reference_{field}.npy")
    result = np.load(case / f"subject_{field}.npy")

    assert result.min() > min(K_MIN, EPSILON_MIN), (
        f"sub_floor: {field} was clipped to the floor instead of repaired (min {result.min():g})"
    )
    np.testing.assert_allclose(
        result,
        reference,
        rtol=RTOL,
        err_msg=f"sub_floor: {field} differs from pybFoam after one correct step",
    )
