# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MULES alpha-advection in every regime the damBreak tutorial does *not* use.

``test_damBreak_comparison.py`` pins the Python MULES transcription against
native interFoam for exactly one set of controls — the tutorial's
``nAlphaCorr 2; nAlphaSubCycles 1; cAlpha 1; MULESCorr yes;``. This module
covers the rest of ``models/mules.py``'s configuration space, one
``parametrize`` entry per regime (TEST_STYLE rule 10):

* ``MULESCorr no``      — the explicit ``MULES::explicitSolve`` branch, never
  taken by the tutorial (which always runs the semi-implicit predictor).
* ``nAlphaSubCycles 2`` — ``alphaEqnSubCycle.H``: the alpha equation is
  advanced twice per time step at ``deltaT/2`` and ``rhoPhi`` is the
  sub-cycle-weighted average of the sub-step mass fluxes.
* ``nAlphaCorr 1``      — a single corrector pass (the ``aCorr > 0``
  under-relaxation branch is then never entered).
* ``cAlpha 0`` / ``cAlpha 4`` — no interface compression at all, and four
  times the tutorial's; ``cAlpha 0`` is the sharpest of the two because the
  expected behaviour is exactly "the ``phir`` term contributes nothing".
* ``alphaApplyPrevCorr yes`` — the ``talphaPhi1Corr0`` cache: the compression
  correction one MULESCorr pass applied is re-applied (through ``MULES::correct``)
  as the *next* pass's seed. State that survives between passes, so it is the
  one regime the first time step cannot show.

**Oracle.** Each regime is written into a copy of ``tutorials/damBreak`` with
pybFoam's own dictionary writer and run twice: once with incompressibleVoF and
once with native interFoam reading the *same* dictionaries. Both solvers share
the CFL routines and therefore the adaptive-dt sequence, so — as in
``test_damBreak_comparison.py`` — agreement is expected to machine precision
(``rtol = atol = 1e-10``), not to a modelling tolerance.

**Horizon.** ``endTime = 0.05 s`` is ~7 adaptive steps of the damBreak column
collapse — long enough that the regimes have stopped being interchangeable, so
"matches interFoam" is a statement about the regime and not about a knob nobody
read. Measured against the tutorial default at 0.05 s (native interFoam, same
dictionaries), ``max |Δalpha.water|`` is 0.232 for ``MULESCorr no``, 0.053 for
``nAlphaSubCycles 2``, 0.158 for ``nAlphaCorr 1``, 0.233 for ``cAlpha 0``,
0.256 for ``cAlpha 4`` and 0.044 for ``alphaApplyPrevCorr yes`` — i.e. every
regime moves the interface by O(0.1), eight to nine orders above the 1e-10
comparison tolerance. A single step would not do: the compression flux and the
sub-cycle weighting act through the accumulated ``alphaPhi10``/``rhoPhi``, which
only separate once the interface has moved — and ``alphaApplyPrevCorr`` has no
cached correction to apply at all until the second pass.

Both solver runs go through subprocesses (``_mules_regime_worker.py`` and the
``interFoam`` binary): a process may construct exactly one ``Foam::Time`` and
this module runs six regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

import numpy as np
import pytest

from ..incompressibleFluid.comparison_helpers import (
    compare_solver_fields,
    get_time_directories,
    read_internal_fields,
)
from .comparison_helpers import FIELDS_TO_COMPARE, run_dambreak_regime

AlphaControls = Mapping[str, Union[bool, float]]

# The tutorial defaults are nAlphaCorr 2, nAlphaSubCycles 1, cAlpha 1,
# MULESCorr yes; every entry below changes exactly one of them.
REGIMES = [
    pytest.param({"MULESCorr": False}, id="MULESCorr_off"),
    pytest.param({"nAlphaSubCycles": 2}, id="nAlphaSubCycles_2"),
    pytest.param({"nAlphaCorr": 1}, id="nAlphaCorr_1"),
    pytest.param({"cAlpha": 0.0}, id="cAlpha_0"),
    pytest.param({"cAlpha": 4.0}, id="cAlpha_4"),
    pytest.param({"alphaApplyPrevCorr": True}, id="alphaApplyPrevCorr_on"),
]


@dataclass
class RegimeRun:
    """One regime, run to ``endTime`` by both solvers; the cases stay on disk."""

    controls: AlphaControls
    python_case: Path
    native_case: Path


@pytest.fixture(scope="module", params=REGIMES)
def regime_run(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> RegimeRun:
    """Run one regime with both solvers, once, for all assertions below."""
    controls: AlphaControls = request.param
    root = tmp_path_factory.mktemp("mulesRegime")
    python_case = root / "incompressibleVoF"
    native_case = root / "interFoam"
    run_dambreak_regime(controls, python_case, native_case)
    return RegimeRun(controls, python_case, native_case)


def test_mules_regime_matches_native_interFoam(regime_run: RegimeRun) -> None:
    """Python MULES reproduces interFoam bit-for-bit in this regime."""
    all_match, failed_fields, failed_details = compare_solver_fields(
        regime_run.python_case,
        regime_run.native_case,
        FIELDS_TO_COMPARE,
        rtol=1e-10,
        atol=1e-10,
    )

    assert all_match, f"regime {dict(regime_run.controls)}: " + ", ".join(
        f"{name}(abs={failed_details[name][0]:.3e}, rel={failed_details[name][1]:.3e})"
        for name in failed_fields
    )


def test_mules_regime_keeps_alpha_water_bounded(regime_run: RegimeRun) -> None:
    """alpha.water stays in [0, 1] — the whole point of the MULES limiter.

    The slack is 1e-4 of the [0, 1] range. MULES is not an exact projection:
    ``MULES::limiter`` is a fixed-point sweep run ``nLimiterIter`` times (5 in
    the tutorial's fvSolution), so what it leaves is the un-converged remainder,
    not round-off — machine precision is the wrong scale here. 1e-4 is two to
    four orders above the largest excursion any of these regimes actually
    produces (1e-5 at ``cAlpha 4``, the sharpest compression) and four orders
    below the O(0.1) over/undershoot an *unlimited* vanLeer + compression
    advection of this collapsing column gives, so it still fails loudly if the
    limiter stops doing its job.
    """
    final_time = get_time_directories(regime_run.python_case)[-1]
    alpha = read_internal_fields(regime_run.python_case, final_time, ["alpha.water"])["alpha.water"]
    out_of_range = (
        f"regime {dict(regime_run.controls)}: alpha.water out of [0, 1] "
        f"(min={np.min(alpha):.6e}, max={np.max(alpha):.6e})"
    )

    assert np.min(alpha) >= -1e-4, out_of_range
    assert np.max(alpha) <= 1.0 + 1e-4, out_of_range
