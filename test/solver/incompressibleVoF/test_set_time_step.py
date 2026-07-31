# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for ``set_time_step`` — interFoam's ``setInitialDeltaT.H`` +``setDeltaT.H``
dual Courant/alpha-Courant adaptive time step, transcribed in ``incompressibleVoF.py``.

Every scenario runs in ``_set_time_step_worker.py``: one ``Foam::Time`` (and one
mesh) per process is a hard OpenFOAM constraint, so all scenarios share the one
``row4`` mesh already used by ``test_alpha_courant.py`` (four cells, each volume
exactly 0.25, x-face area exactly 1) and are evaluated in a single worker pass.

Each scenario seeds a **uniform** ``U = (u_x, 0, 0)`` and a **uniform**
``alpha.water = 0.5`` (squarely inside the ``0.01 <= alpha <= 0.99`` interface
band for every cell). With a uniform field every cell has the same
``sumPhi = 2*|u_x|`` (two x-faces, boundary duplicated by the ``zeroGradient``
U condition), so the flow Courant number and the interface Courant number
collapse to the *same* exact number:

    CoNum = alphaCoNum = 0.5 * (2*|u_x| / 0.25) * dt0 = 4*|u_x|*dt0

(confirmed against the real ``computeCFLNumber``/``compute_alpha_courant_number``
on this mesh: e.g. ``u_x=1, dt0=0.25`` gives exactly ``1.0`` for both). That
collapses ``setDeltaT.H``'s dual limiter to one hand-computable number per
scenario, so ``maxCo``/``maxAlphaCo``/``maxDeltaT`` can be driven independently
through real ``system/controlDict`` variants (via pybFoam's own
``dictionary.read``/``set``/``write``) while every expected ``deltaT`` below is
a pre-written literal derived by hand from ``setDeltaT.H``:

    maxDeltaTFact = min(maxCo/(CoNum+1e-15), maxAlphaCo/(alphaCoNum+1e-15))
    deltaTFact    = min(min(maxDeltaTFact, 1 + 0.1*maxDeltaTFact), 1.2)
    newDeltaT     = min(deltaTFact * dt0, maxDeltaT)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from numpy.testing import assert_allclose

_HERE = Path(__file__).parent
_CASES = _HERE / "cases" / "alphaCourant"
_WORKER = _HERE / "_set_time_step_worker.py"

# Every scenario shares the row4 mesh, a uniform U=(u_x,0,0) and a uniform
# alpha=0.5 (all four cells inside the interface band) -> CoNum = alphaCoNum
# = 4*|u_x|*dt0 exactly. "control_dict" carries only the keys that must be
# *present*; an omitted key exercises the "absent from controlDict" default.
SCENARIOS: dict[str, dict[str, Any]] = {
    # 1. adjustTimeStep no / absent -> early return, dt unchanged, setDeltaT
    # never called. maxCo/maxAlphaCo/maxDeltaT are irrelevant (short-circuit).
    "adjust_no": {
        "u_x": 1.0,
        "alpha": 0.5,
        "dt0": 0.25,
        "control_dict": {
            "adjustTimeStep": False,
            "maxCo": 0.1,
            "maxAlphaCo": 0.1,
            "maxDeltaT": 0.1,
        },
    },
    "adjust_absent": {
        "u_x": 1.0,
        "alpha": 0.5,
        "dt0": 0.25,
        "control_dict": {},  # no adjustTimeStep key at all
    },
    # 2a. Flow Courant is the limiter: maxCo tight, maxAlphaCo loose.
    # CoNum = alphaCoNum = 4*1.0*0.25 = 1.0.
    # maxDeltaTFact = min(0.1/1.0, 10.0/1.0) = 0.1 (flow branch binds).
    # deltaTFact = min(min(0.1, 1+0.1*0.1=1.01), 1.2) = 0.1.
    # newDeltaT = min(0.1*0.25, 1.0) = 0.025.
    "flow_limits": {
        "u_x": 1.0,
        "alpha": 0.5,
        "dt0": 0.25,
        "control_dict": {
            "adjustTimeStep": True,
            "maxCo": 0.1,
            "maxAlphaCo": 10.0,
            "maxDeltaT": 1.0,
        },
    },
    # 2b. Interface (alpha) Courant is the limiter: maxAlphaCo tight, maxCo loose.
    # CoNum = alphaCoNum = 1.0.
    # maxDeltaTFact = min(10.0/1.0, 0.2/1.0) = 0.2 (alpha branch binds).
    # deltaTFact = min(min(0.2, 1+0.1*0.2=1.02), 1.2) = 0.2.
    # newDeltaT = min(0.2*0.25, 1.0) = 0.05.
    "alpha_limits": {
        "u_x": 1.0,
        "alpha": 0.5,
        "dt0": 0.25,
        "control_dict": {
            "adjustTimeStep": True,
            "maxCo": 10.0,
            "maxAlphaCo": 0.2,
            "maxDeltaT": 1.0,
        },
    },
    # 3. Ramp cap: a tiny Courant number makes maxDeltaTFact huge, but the
    # ramp caps the per-step growth at 1.2x regardless.
    # CoNum = alphaCoNum = 4*1.0*1e-6 = 4e-6.
    # maxDeltaTFact = min(1.0/4e-6, 1.0/4e-6) ~ 250000 (the 1e-15 epsilon is
    # negligible: 1e-15/4e-6 ~ 2.5e-10).
    # deltaTFact = min(min(250000, 1+0.1*250000=25001), 1.2) = 1.2.
    # newDeltaT = min(1.2*1e-6, 1.0) = 1.2e-6.
    "ramp_cap": {
        "u_x": 1.0,
        "alpha": 0.5,
        "dt0": 1e-6,
        "control_dict": {
            "adjustTimeStep": True,
            "maxCo": 1.0,
            "maxAlphaCo": 1.0,
            "maxDeltaT": 1.0,
        },
    },
    # 4. maxDeltaT clamp: the Courant limit alone would allow dt0*1.2, but
    # maxDeltaT caps it lower.
    # CoNum = alphaCoNum = 4*0.01*1.0 = 0.04.
    # maxDeltaTFact = min(20.0/0.04, 20.0/0.04) = 500.
    # deltaTFact = min(min(500, 1+0.1*500=51), 1.2) = 1.2 (ramp-capped, not
    # Courant-limited -> without the clamp this would give 1.2*1.0 = 1.2).
    # newDeltaT = min(1.2*1.0, 0.05) = 0.05 (the maxDeltaT clamp binds).
    "max_delta_t_clamp": {
        "u_x": 0.01,
        "alpha": 0.5,
        "dt0": 1.0,
        "control_dict": {
            "adjustTimeStep": True,
            "maxCo": 20.0,
            "maxAlphaCo": 20.0,
            "maxDeltaT": 0.05,
        },
    },
    # 5a. maxCo/maxAlphaCo absent -> interFoam default 1.0 for both. Chosen so
    # neither the ramp cap nor maxDeltaT clamp engages, isolating the default:
    # CoNum = alphaCoNum = 4*2.0*0.25 = 2.0.
    # maxDeltaTFact = min(1.0/2.0, 1.0/2.0) = 0.5 (uses the 1.0 default).
    # deltaTFact = min(min(0.5, 1+0.1*0.5=1.05), 1.2) = 0.5.
    # newDeltaT = min(0.5*0.25, 1.0) = 0.125.
    "defaults_co": {
        "u_x": 2.0,
        "alpha": 0.5,
        "dt0": 0.25,
        "control_dict": {"adjustTimeStep": True, "maxDeltaT": 1.0},
    },
    # 5b. maxCo/maxAlphaCo/maxDeltaT all absent -> defaults 1.0/1.0/1.0. Chosen
    # so the ramp cap engages (Courant not limiting) and the maxDeltaT default
    # is what actually clamps the result (distinguishing 1.0 from "no clamp"):
    # CoNum = alphaCoNum = 4*0.01*1.0 = 0.04.
    # maxDeltaTFact = min(1.0/0.04, 1.0/0.04) = 25 (uses the 1.0 default).
    # deltaTFact = min(min(25, 1+0.1*25=3.5), 1.2) = 1.2.
    # newDeltaT = min(1.2*1.0, 1.0) = 1.0 (the 1.0 maxDeltaT default clamps it;
    # without the clamp this would be 1.2).
    "default_max_delta_t": {
        "u_x": 0.01,
        "alpha": 0.5,
        "dt0": 1.0,
        "control_dict": {"adjustTimeStep": True},
    },
    # 6. Mid-run step ("advance": the worker increments the Foam::Time first, so
    # timeIndex() != 0). Same controlDict as 2a, so the deltaT is the same 0.025 —
    # what differs is that setInitialDeltaT.H is gated off and setDeltaT is called
    # once instead of twice. MUST stay last: the increment is not undone.
    "flow_limits_mid_run": {
        "u_x": 1.0,
        "alpha": 0.5,
        "dt0": 0.25,
        "advance": True,
        "control_dict": {
            "adjustTimeStep": True,
            "maxCo": 0.1,
            "maxAlphaCo": 10.0,
            "maxDeltaT": 1.0,
        },
    },
}


def _stage(dest: Path) -> Path:
    """Copy the checked-in row4 case inputs; the worker meshes on top."""
    shutil.copytree(_CASES / "common", dest)
    shutil.copy(_CASES / "row4" / "blockMeshDict", dest / "system")
    shutil.copytree(dest / "0.orig", dest / "0")
    # Pristine copy of the base controlDict: the worker rewrites
    # system/controlDict from this template every scenario, so no scenario's
    # keys leak into the next one.
    shutil.copy(dest / "system" / "controlDict", dest / "controlDict.template")
    return dest


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run every scenario in SCENARIOS on the row4 mesh, in one worker process."""
    case = _stage(tmp_path_factory.mktemp("row4") / "case")
    request = case / "request.json"
    request.write_text(json.dumps({"scenarios": SCENARIOS}))
    subprocess.run(
        [sys.executable, str(_WORKER), str(case), str(request)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(json.loads((case / "result.json").read_text()))


# --- 1. adjustTimeStep no / absent: early return -----------------------------


def test_adjust_time_step_no_leaves_delta_t_unchanged(results: dict[str, Any]) -> None:
    scenario = results["adjust_no"]
    assert_allclose(
        scenario["final_dt"],
        0.25,
        rtol=1e-12,
        err_msg="adjustTimeStep=no must leave deltaT untouched",
    )
    assert scenario["set_delta_t_calls"] == [], "setDeltaT must never be called"


def test_adjust_time_step_absent_key_leaves_delta_t_unchanged(
    results: dict[str, Any],
) -> None:
    scenario = results["adjust_absent"]
    assert_allclose(
        scenario["final_dt"],
        0.25,
        rtol=1e-12,
        err_msg="adjustTimeStep absent (default False) must leave deltaT untouched",
    )
    assert scenario["set_delta_t_calls"] == [], "setDeltaT must never be called"


# --- 2. dual limiter: each branch binds in turn ------------------------------


def test_flow_courant_number_is_the_limiter(results: dict[str, Any]) -> None:
    # maxCo=0.1 (tight) vs maxAlphaCo=10.0 (loose): the flow-CFL ratio is the
    # smaller of the two, so it alone sets deltaTFact.
    assert_allclose(
        results["flow_limits"]["final_dt"],
        0.025,
        rtol=1e-9,
        err_msg="flow Courant number must be the binding constraint",
    )


def test_alpha_courant_number_is_the_limiter(results: dict[str, Any]) -> None:
    # maxAlphaCo=0.2 (tight) vs maxCo=10.0 (loose): the interface-CFL ratio is
    # the smaller of the two, so it alone sets deltaTFact.
    assert_allclose(
        results["alpha_limits"]["final_dt"],
        0.05,
        rtol=1e-9,
        err_msg="interface (alpha) Courant number must be the binding constraint",
    )


# --- 3. ramp cap: at most 1.2x growth per step -------------------------------


def test_ramp_caps_growth_at_1_2x_per_step(results: dict[str, Any]) -> None:
    # dt0 = 1e-6; a vanishingly small Courant number would otherwise let
    # maxDeltaTFact run into the tens of thousands, but the ramp still caps
    # growth at exactly 1.2x the previous dt: 1.2 * 1e-6 = 1.2e-6.
    assert_allclose(
        results["ramp_cap"]["final_dt"],
        1.2e-6,
        rtol=1e-6,
        err_msg="a vanishingly small Courant number must still cap growth at 1.2x",
    )


# --- 4. maxDeltaT clamp -------------------------------------------------------


def test_max_delta_t_clamps_even_when_courant_would_allow_more(
    results: dict[str, Any],
) -> None:
    scenario = results["max_delta_t_clamp"]
    # Without the maxDeltaT=0.05 clamp, the ramp-capped factor (1.2) would
    # give 1.2*1.0 = 1.2 -> the clamp is what pins it at 0.05.
    assert_allclose(
        scenario["final_dt"],
        0.05,
        rtol=1e-9,
        err_msg="maxDeltaT must clamp deltaT even though the Courant limit allows more",
    )


# --- 5. defaults when keys are absent from controlDict -----------------------


def test_default_max_co_and_max_alpha_co_are_one_when_absent(
    results: dict[str, Any],
) -> None:
    assert_allclose(
        results["defaults_co"]["final_dt"],
        0.125,
        rtol=1e-9,
        err_msg="maxCo/maxAlphaCo must default to 1.0 when absent from controlDict",
    )


def test_default_max_delta_t_is_one_when_absent(results: dict[str, Any]) -> None:
    assert_allclose(
        results["default_max_delta_t"]["final_dt"],
        1.0,
        rtol=1e-9,
        err_msg="maxDeltaT must default to 1.0 when absent from controlDict",
    )


# --- 6. setInitialDeltaT.H runs once, on the first step only ------------------


def test_first_step_runs_the_initial_pass_before_the_damped_one(
    results: dict[str, Any],
) -> None:
    # interFoam runs setInitialDeltaT.H (with its own CourantNo.H) before the loop,
    # so the first pass calls setDeltaT twice. The two values coincide on this case
    # because setDeltaT.H's reduction is immediate too; what the extra call buys is
    # the Time::adjustDeltaT it triggers, which an adjustableRunTime case then grows
    # from (this mesh's controlDict is writeControl runTime, so nothing snaps here).
    calls = results["flow_limits"]["set_delta_t_calls"]
    assert len(calls) == 2, "setInitialDeltaT.H must precede setDeltaT.H on step 1"
    assert_allclose(
        calls[0],
        0.025,  # min(maxCo*dt0/CoNum, min(dt0, maxDeltaT)) = min(0.025, 0.25)
        rtol=1e-9,
        err_msg="the initial pass must apply the CFL limit undamped",
    )


def test_mid_run_step_skips_the_initial_pass(results: dict[str, Any]) -> None:
    scenario = results["flow_limits_mid_run"]
    assert scenario["set_delta_t_calls"] == pytest.approx([0.025]), (
        "setInitialDeltaT.H is gated on timeIndex() == 0 and must not run again"
    )
    assert_allclose(
        scenario["final_dt"],
        0.025,
        rtol=1e-9,
        err_msg="the mid-run step must reach the same deltaT as the first one",
    )
