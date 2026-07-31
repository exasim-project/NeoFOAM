# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity of the NeoN closures' **wall functions** against pybFoam, in the wall cells.

:mod:`test_neon_turbulence_parity` deliberately runs on a wall-free box, so the
epsilon / omega / kqR / nutk wall functions and their model-side halves (the
near-wall production override and the near-wall matrix-cell pin) are never
exercised there. This module is that missing half: one ``correct`` step on
``wall_function_base`` — a cube whose **four** x/z faces are ``wall`` — asserted
first over the wall cells alone, so a wall-treatment regression names itself
rather than surfacing as a whole-field mismatch, and then over the whole field.

Three properties of the case make it discriminating where a stock walled box is not:

* **Corner weighting.** The 16 cells along the four x/z edges each own *two* wall
  faces, so ``epsilon0`` / ``omega0`` and the near-wall ``G`` are the
  ``1/nWallFaces``-weighted sums OpenFOAM accumulates, not a single face value.
* **Both branches of the STEPWISE gate.** ``nu = 2e-3`` and a random ``k`` over
  ``[0.02, 0.25]`` put the wall-face ``y+ = Cmu^0.25 y sqrt(k)/nu`` on both sides
  of ``yPlusLam = 11.53``, so ``nutkWallFunction`` takes its viscous branch
  (``nut_w = 0``) on some faces and its log branch on others within one run. A
  uniform ``k`` — every other walled case here — puts every face on one side.
* **A conservative flux.** ``U = (0, 2x + 1.5z, 0)`` has no wall-normal component
  on any wall patch, so ``createPhi(U)`` is divergence-free to round-off. That
  keeps the dilatation terms both backends carry (``fvm::SuSp((2/3) divU, k)``,
  ``kEpsilon.C:280``) identically zero here, so this module measures the wall
  functions alone; the dilatation terms have their own case and module
  (:mod:`test_dilatation_parity`, a non-conservative flux on the same box).

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
_CASE = _HERE / "wall_function_base"
_MODELS = _HERE / "parity_models"

#: ``(model, fields it owns, agreement bound as a fraction of the field's peak)``.
#: Both models use ``kqRWallFunction`` for ``k`` and ``nutkWallFunction`` for
#: ``nut``; they differ in the second transport unknown and hence in the wall
#: function (``epsilonWallFunction`` vs ``omegaWallFunction``) and the matrix pin
#: under test. ``SpalartAllmaras`` is absent: it needs
#: ``nutUSpaldingWallFunction``, i.e. a different ``0/nut``, so it belongs in its
#: own case rather than this overlay.
#:
#: Both backends solve the same matrix to an absolute residual of 1e-14
#: (``wall_function_base``'s ``fvSolution``), so the bound is round-off, not
#: solver tolerance — ``kEpsilon`` measures 4e-14 of peak. ``kOmegaSST`` measures
#: 4e-8: its ``F1``/``F2`` blending chain (``tanh``/``pow4``/``sqrt`` over
#: ``CDkOmega``) is evaluated in a different association order on the two
#: backends, the same round-off the steady solver comparison budgets 3e-5 for.
#: Each bound is ~25x its measured value, and both are orders below every
#: wall-treatment bug this class of test has caught (all >= 1e-3).
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


#: The wall cells of ``wall_function_base``. blockMesh numbers a single hex block
#: x-fastest, so cell ``i + 4j + 16k`` sits at column ``i`` / row ``j`` / layer
#: ``k``; a cell touches a wall exactly when it is in the first or last x column
#: or z layer. 48 of the 64 cells, of which the 16 with both are the edge cells
#: that carry two wall faces.
WALL_CELLS = np.array(
    [
        i + 4 * j + 16 * k
        for k in range(4)
        for j in range(4)
        for i in range(4)
        if i in (0, 3) or k in (0, 3)
    ]
)


@pytest.mark.parametrize(("model", "fields", "bound"), CASES, ids=[c[0] for c in CASES])
def test_wall_cells_match_pybfoam_after_one_correct(
    model: str, fields: tuple[str, ...], bound: float, tmp_path: Path
) -> None:
    """One ``correct`` step leaves the NeoN wall cells equal to pybFoam's."""
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)
    shutil.copyfile(
        _MODELS / model / "turbulenceProperties",
        case / "constant" / "turbulenceProperties",
    )

    _run_worker("mesh", case)  # the case ships its own 0/ fields; do not reseed
    _run_worker("reference", case)  # pybFoam wall functions
    _run_worker("subject", case)  # the NeoN closure's wall functions

    for name in fields:
        reference = np.load(case / f"reference_{name}.npy")
        result = np.load(case / f"subject_{name}.npy")
        peak = float(np.max(np.abs(reference))) or 1.0
        np.testing.assert_allclose(
            result[WALL_CELLS],
            reference[WALL_CELLS],
            rtol=0.0,
            atol=bound * peak,
            err_msg=f"{model}: {name} differs in the wall cells after one correct step",
        )
        np.testing.assert_allclose(
            result,
            reference,
            rtol=0.0,
            atol=bound * peak,
            err_msg=f"{model}: {name} differs after one correct step",
        )
