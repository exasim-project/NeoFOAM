# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NOTE: no `from __future__ import annotations` — the loop operation's interface
# annotations must stay live for the dependency resolver.

"""Native-parity spec for the adaptive ``deltaT`` trajectory.

The step controller has to reproduce the *sequence*
``setInitialDeltaT.H`` -> ``setDeltaT.H`` -> ``Foam::Time::adjustDeltaT`` exactly.
A trajectory that merely reaches the same ``endTime`` by different steps
perturbs every field, and the drop-in study compares at ``rtol = 1e-10``, so it
cannot hide there — it has to be pinned here.

The table is driven through the *real* ``courant`` / ``maxDeltaT`` models and the
real gather points the solver wires (``initialTimeStepConstraint``,
``timeStepConstraint``, ``maxTimeStep``), so a deviation in either a contribution
formula or the loop's damping is caught. Every expectation was computed by hand
from the three OpenFOAM sources, never from the neofoam implementation; the
9-step trajectory additionally reproduces the ``pimpleFoam/RAS/TJunction``
reference log (``deltaT = 0.00117647``, then ``0.00144796``, ``Time = 0.00262443``).

``computeCFLNumber`` is the single stubbed seam: a row states ``coPerDeltaT``,
the mesh/flux quantity, and the Courant number is ``coPerDeltaT * deltaT`` —
exactly the ``CourantNo.H`` relation (it multiplies that quantity by
``runTime.deltaTValue()``), which lets a row name a flow condition without a mesh.

Tolerance: ``rel = 1e-12``. The expectations are exact IEEE doubles apart from
OpenFOAM's ``SMALL`` denominator epsilon in ``maxCo/(CoNum + SMALL)``, which
perturbs the growth factor by ~1e-15 relative.
"""

import importlib
from typing import Any, Optional

import pytest

from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.solution_loop import (
    SolutionLoop,
    set_time_step,
    solutionLoop,
)
from neofoam.framework.context import Context
from neofoam.framework.dependency_resolver import (
    DependencyResolver,
    wrap_with_dependency_resolution,
)
from neofoam.framework.model import ModelRuntime
from neofoam.solver.incompressibleFluid.models.courant import CourantConfig, courant
from neofoam.solver.incompressibleFluid.models.max_delta_t import (
    MaxDeltaTConfig,
    maxDeltaT,
)

courant_mod = importlib.import_module("neofoam.solver.incompressibleFluid.models.courant")

# The expectations are exact doubles except for the SMALL denominator epsilon.
NATIVE_RTOL = 1e-12


def _loop(
    *,
    deltaT: float,
    writeControl: str = "timeStep",
    writeInterval: float = 1.0,
    endTime: float = 1.0,
) -> SolutionLoop:
    return SolutionLoop(
        state=LoopState(
            value=0.0,
            delta_t=deltaT,
            end_time=endTime,
            write_control=writeControl,
            write_interval=writeInterval,
        )
    )


def _contributors(*, maxCo: Optional[float], maxDeltaTValue: Optional[float]) -> list[ModelRuntime]:
    """The model runtimes a case with these controlDict entries would detect.

    ``maxCo`` absent stands for ``adjustTimeStep no``: neither model is detected,
    so the loop sees no opinion at all and OpenFOAM never calls ``setDeltaT``.
    """
    runtimes: list[ModelRuntime] = []
    if maxCo is not None:
        runtimes.append(
            ModelRuntime(spec=courant, name="courant", config=CourantConfig(maxCo=maxCo))
        )
    if maxDeltaTValue is not None:
        runtimes.append(
            ModelRuntime(
                spec=maxDeltaT,
                name="maxDeltaT",
                config=MaxDeltaTConfig(maxDeltaT=maxDeltaTValue),
            )
        )
    return runtimes


def _step(loop: SolutionLoop, contributors: list[ModelRuntime]) -> None:
    """One ``set_time_step`` pass, wired the way ``create_fields`` wires it."""
    loop_rt = ModelRuntime(spec=solutionLoop, name="solutionLoop", config=None)
    ctx = Context(
        fields={"phi": object()},
        models={
            "solution_loop": loop,
            "solutionLoop": loop_rt,
            **{rt.name: rt for rt in contributors},
        },
    )
    wrap_with_dependency_resolution(
        set_time_step, instance=None, dependency_resolver=DependencyResolver()
    )(ctx)


def _stub_courant(
    monkeypatch: pytest.MonkeyPatch, loop: SolutionLoop, co_per_delta_t: float
) -> None:
    """``CourantNo.H``: Co = (mesh/flux quantity) * the live deltaT."""
    monkeypatch.setattr(
        courant_mod,
        "computeCFLNumber",
        lambda phi: (co_per_delta_t * loop.current_delta_t(), 0.0),
    )


# --- setInitialDeltaT.H + setDeltaT.H + adjustDeltaT, one step ------------
#
# Each row is one controlDict; `expected` is the deltaT the OpenFOAM sources
# produce for the first loop pass (t = 0, timeIndex = 0).
_ROWS: list[tuple[dict[str, Any], float]] = [
    # setDeltaT.H uses maxCo/(CoNum + SMALL) as a denominator epsilon, never as a
    # cut-off: a quiescent start grows by the full 1.2 cap (it does NOT freeze).
    ({"coPerDeltaT": 0.0, "maxCo": 5.0}, 0.0012),
    # "Reduction of time-step is immediate": factor 0.5 applied in full.
    ({"coPerDeltaT": 10000.0, "maxCo": 5.0}, 0.0005),
    # 1.111 < maxCo/Co < 2 -> the 1 + 0.1*fact damping term binds.
    ({"coPerDeltaT": 1000.0, "maxCo": 1.5}, 0.00115),
    # maxCo/Co = 3 -> the hard 1.2 growth cap binds.
    ({"coPerDeltaT": 1000.0, "maxCo": 3.0}, 0.0012),
    # maxDeltaT is applied AFTER the damping, not folded into the damped factor:
    # fact = min(1.2, 1.12, 1.2) = 1.12 -> 0.00112, which is below the 0.00115 cap.
    # Damping the cap instead would give min(1.15, 1.115, 1.2)*dt = 0.001115.
    ({"coPerDeltaT": 1000.0, "maxCo": 1.2, "maxDeltaTValue": 0.00115}, 0.00112),
    # ... and when the cap really is the binding constraint it clips hard.
    ({"coPerDeltaT": 0.0, "maxCo": 5.0, "maxDeltaTValue": 0.0011}, 0.0011),
    # adjustableRunTime: 1.2*0.001 = 0.0012 snapped onto 0.02/17 (TJunction step 1).
    (
        {
            "coPerDeltaT": 0.0,
            "maxCo": 5.0,
            "writeControl": "adjustableRunTime",
            "writeInterval": 0.02,
        },
        0.0011764705882352942,
    ),
    # setInitialDeltaT.H fires (CoNum = 1e-3 > SMALL). It cannot raise the step, but
    # its setDeltaT call snaps 0.001 onto 0.0044/4 = 0.0011 first, so the 1.2 growth
    # then snaps onto 0.0044/3. Skipping the initial pass lands on 0.0011 instead.
    (
        {
            "coPerDeltaT": 1.0,
            "maxCo": 5.0,
            "writeControl": "adjustableRunTime",
            "writeInterval": 0.0044,
        },
        0.0014666666666666667,
    ),
    # Same controlDict but quiescent: CoNum <= SMALL, so setInitialDeltaT.H skips
    # its setDeltaT (and its snap) entirely.
    (
        {
            "coPerDeltaT": 0.0,
            "maxCo": 5.0,
            "writeControl": "adjustableRunTime",
            "writeInterval": 0.0044,
        },
        0.0011,
    ),
    # adjustTimeStep no: setDeltaT.H is never entered, so Time::adjustDeltaT is
    # never reached and the fixed step survives an adjustableRunTime writeControl.
    (
        {"coPerDeltaT": 0.0, "writeControl": "adjustableRunTime", "writeInterval": 0.0044},
        0.001,
    ),
]

_IDS = [
    "quiescent_grows_by_the_1p2_cap",
    "supercritical_shrinks_at_once",
    "growth_damped_by_one_plus_tenth_fact",
    "growth_clamped_at_1p2",
    "max_delta_t_clips_after_the_damping",
    "max_delta_t_clips_the_1p2_growth",
    "adjustable_write_snaps_the_grown_step",
    "initial_pass_snaps_before_the_first_growth",
    "quiescent_start_skips_the_initial_pass",
    "fixed_step_is_never_snapped",
]


@pytest.mark.parametrize(("row", "expected"), _ROWS, ids=_IDS)
def test_first_step_matches_the_openfoam_sources(
    row: dict[str, Any], expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _loop(
        deltaT=0.001,
        writeControl=row.get("writeControl", "timeStep"),
        writeInterval=row.get("writeInterval", 1.0),
    )
    _stub_courant(monkeypatch, loop, row["coPerDeltaT"])

    _step(loop, _contributors(maxCo=row.get("maxCo"), maxDeltaTValue=row.get("maxDeltaTValue")))

    assert loop.state.delta_t == pytest.approx(expected, rel=NATIVE_RTOL), (
        f"deltaT trajectory diverges from setDeltaT.H for controlDict {row}"
    )


# --- the whole run: pimpleFoam/RAS/TJunction ------------------------------
#
# deltaT 0.001, maxCo 5, endTime 0.02, writeControl adjustable / writeInterval
# 0.02, quiescent start. maxCo = 5 keeps the Courant factor far from binding, so
# the trajectory is the 1.2 cap against the write-time snapping. The first two
# entries and the third time value are read off the native log.
_TJUNCTION_DELTA_T = [
    0.0011764705882352942,
    0.0014479638009049776,
    0.001737556561085973,
    0.0019547511312217195,
    0.0022805429864253394,
    0.002850678733031674,
    0.002850678733031674,
    0.002850678733031674,
    0.002850678733031676,
]


def test_tjunction_trajectory_matches_the_native_log(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = _loop(
        deltaT=0.001,
        writeControl="adjustableRunTime",
        writeInterval=0.02,
        endTime=0.02,
    )
    _stub_courant(monkeypatch, loop, 0.0)
    contributors = _contributors(maxCo=5.0, maxDeltaTValue=None)

    trajectory: list[float] = []
    while loop.running():
        _step(loop, contributors)
        trajectory.append(loop.state.delta_t)
        loop.advance()

    assert trajectory == pytest.approx(_TJUNCTION_DELTA_T, rel=NATIVE_RTOL), (
        "the TJunction deltaT sequence diverges from the pimpleFoam log"
    )
    # The native run lands exactly on the write time; so must this one.
    assert loop.state.value == pytest.approx(0.02, abs=1e-15)
