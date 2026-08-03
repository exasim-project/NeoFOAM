# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The NeoN ``kEpsilon`` closure survives a step from a degenerate ``k`` / ``epsilon``.

Both scenarios take **one ``correct`` step from the same walled box**, differing
only in the seeded ``0/k`` (and ``0/epsilon``), and both assert the same three
things about the same three fields: the step stays finite, the solved fields
respect the closure's floor, and they still equal what OpenFOAM's own ``kEpsilon``
produces from the identical files. Hence one parametrized body.

The shared case is :func:`_parity_case.walled_case` — a 4 x 4 x 4 box whose two z
patches are walls, with ``kqRWallFunction`` / ``epsilonWallFunction`` / ``nutk``
boundaries and a linear solver at ``tolerance 1e-14, relTol 0``. Each scenario is that
case with its ``0/`` internal fields rewritten by :func:`seeded`:

* **near_zero_k** — ``k`` seeded at ``1e-8`` against the stock ``epsilon = 14.855``,
  i.e. ``epsilon/k ~ 1.5e9``. ``kEpsilon`` divides by ``k`` (the ``C2 epsilon/k``
  dissipation coefficient and the ``epsilon/k`` sink of the k equation) while ``k``
  is only floored at ``kMin = 1e-15`` after each solve, so those ratios reach ~1e9 —
  the arithmetic that precedes an ``FE_OVERFLOW`` (the ``FOAM_SIGFPE`` "Floating
  point exception" the verification sweep hit) once a run starts to diverge. A
  solved field may sit *on* the floor here.

* **sub_floor_k** — the near-zero background with four cells seeded negative
  (:data:`SUB_FLOOR_SEEDS`): two in ``k`` (both interior) and two in ``epsilon`` (one a
  ``zMin`` wall cell, one interior), so the repair is exercised both with and without
  a boundary face in its stencil. ``Foam::bound``
  (``src/finiteVolume/cfdTools/general/bound/bound.C``) does **not** clip a
  non-positive cell to the floor: it gives it ``fvc::average(max(vsf, lower))`` —
  the face-area-weighted average of the floored neighbourhood — and only then takes
  the max against the floor. The distinction is not cosmetic: clipping ``epsilon``
  to ``1e-15`` leaves ``nut = Cmu k^2 / epsilon`` dividing by ``1e-15``, which is how
  a single bad cell turns into ``nut ~ 1e10`` and takes the run with it. So here a
  repaired cell must land *strictly above* the floor — landing on one is the bug
  this scenario exists for. OpenFOAM bounds as it constructs
  (``bound(k_, kMin_); bound(epsilon_, epsilonMin_)``) and so does the NeoN
  closure's ``build``, so the comparison covers the repair *and* everything
  downstream of it.

Discriminating power for the repair: swapping it for the old hard floor
(``field_max(k, kMin)``) moves ``k`` and ``epsilon`` by ~100% of their own value in
the seeded cells and ``nut`` by a factor of ~1e4 — four orders outside that
scenario's tolerance.

One ``Foam::Time`` per process, so the roles run as subprocesses of
:mod:`_parity_worker` (see :mod:`_parity_case`) and hand their fields over as
``.npy``. The three roles are driven **once per scenario** by the module-scoped
:func:`stepped_cases` fixture, not once per compared field.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neofoam.tooling.casebuild import Step, patch
from turbulence._parity_case import run_worker, walled_case

#: The closure's floors (``models.kEpsilon.kMin`` / ``epsilonMin``).
FLOOR = 1e-15

#: ``scenario -> (rtol, a repaired cell must sit strictly above the floor)``.
#:
#: ``near_zero_k``: both solves stop at an absolute residual of 1e-14
#: (the shared ``parity_base/system/fvSolution``), which for a ``k`` of order 1e-9 is a
#: relative slack of ~1e-5; 1e-4 keeps that headroom without hiding a coefficient-
#: or term-level disagreement, which would be O(1) here.
#:
#: ``sub_floor_k``: measured 1.5e-10 across k / epsilon / nut, set by the same
#: 1e-14 absolute tolerance against fields of order 1e-9 .. 1e-4. 1e-8 keeps that
#: headroom and still leaves four orders of margin to the ~100% error the hard-floor
#: repair produces.
SCENARIOS = {
    "near_zero_k": (1.0e-4, False),
    "sub_floor_k": (1.0e-8, True),
}

#: The fields the closure solves plus the eddy viscosity they feed.
FIELDS = ["k", "epsilon", "nut"]

#: The 4 x 4 x 4 cells of the walled case, and the degenerate background each field is
#: seeded at (the stock ``epsilon``, against a ``k`` nine orders below it).
N_CELLS = 64
BACKGROUND = {"k": 1.0e-8, "epsilon": 14.855}

#: ``sub_floor_k``'s negative cells, ``field -> {cell: value}``. blockMesh numbers the
#: single hex block x-fastest, so cell ``i + 4j + 16k`` sits at column ``i`` / row ``j``
#: / layer ``k``: ``epsilon``'s cell 5 is in the ``zMin`` wall layer and its cell 37 is
#: interior, while both of ``k``'s are interior.
SUB_FLOOR_SEEDS = {
    "k": {21: -1.0e-8, 42: -5.0e-9},
    "epsilon": {5: -14.855, 37: -1.0},
}


def seeded(field: str, negatives: dict[int, float]) -> Step:
    """Rewrite ``0/<field>``'s internal field: the background, with *negatives* punched in.

    The boundary conditions the case ships (the wall functions) are left untouched —
    only the internal field is replaced, so the scenarios differ from the base case in
    exactly the cell values named here.
    """
    values = np.full(N_CELLS, BACKGROUND[field])
    for cell, value in negatives.items():
        values[cell] = value
    body = " ".join(f"{v:.16e}" for v in values)
    return patch(f"0/{field}", internalField=f"nonuniform List<scalar> {N_CELLS} ( {body} )")


#: ``scenario -> the 0/ rewrites that turn the walled case into it``.
SEEDING: dict[str, tuple[Step, ...]] = {
    "near_zero_k": (patch("0/k", internalField=f"uniform {BACKGROUND['k']:g}"),),
    "sub_floor_k": tuple(seeded(field, cells) for field, cells in SUB_FLOOR_SEEDS.items()),
}


@pytest.fixture(scope="module")
def stepped_cases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Take the one ``correct`` step per scenario, once, on both backends."""
    cases = {}
    for scenario in SCENARIOS:
        pipeline = walled_case()
        for step in SEEDING[scenario]:
            pipeline = pipeline | step
        case = pipeline.build_at(tmp_path_factory.mktemp(scenario) / "case").path
        run_worker("mesh", case)
        run_worker("reference", case)  # OpenFOAM's kEpsilon, same files
        run_worker("subject", case)  # the NeoN closure
        cases[scenario] = case
    return cases


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("field", FIELDS)
def test_degenerate_step_stays_bounded_and_matches_pybfoam(
    scenario: str, field: str, stepped_cases: dict[str, Path]
) -> None:
    """One ``correct`` step from a degenerate state stays finite, bounded and on parity."""
    rtol, strictly_above_floor = SCENARIOS[scenario]
    case = stepped_cases[scenario]

    reference = np.load(case / f"reference_{field}.npy")
    result = np.load(case / f"subject_{field}.npy")

    assert np.isfinite(result).all(), f"{scenario}: {field} is not finite after one step"
    if strictly_above_floor:
        assert result.min() > FLOOR, (
            f"{scenario}: {field} was clipped to the floor instead of repaired "
            f"(min {result.min():g})"
        )
    else:
        assert result.min() >= FLOOR, (
            f"{scenario}: {field} fell below the closure's floor (min {result.min():g})"
        )
    np.testing.assert_allclose(
        result,
        reference,
        rtol=rtol,
        err_msg=f"{scenario}: {field} differs from pybFoam after one correct step",
    )
