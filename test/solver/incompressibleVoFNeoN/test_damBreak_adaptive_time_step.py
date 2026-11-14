# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoFNeoN with interFoam's adaptive time step against native interFoam, per executor.

The case is ``test_damBreak_comparison``'s ``commonBCs`` variant (sigma 0, diagonal p_rgh
preconditioner, the atmosphere patch types both codes evaluate identically) with the
tutorial's own ``adjustTimeStep yes``, ``maxCo 1``, ``maxAlphaCo 1``, ``maxDeltaT 1`` and
``writeControl adjustable``, run to ``t = 0.1`` (26 steps) with a write every 0.025. The
1.2 growth cap and the write-time snapping set the step up to ``t = 0.06``, the Courant
limit after that (CoNum reaches 1.6 once, and the step is halved). With a write every 0.01
the snapping alone would hold deltaT at 1/300 and the Courant limit would never act.
``writePrecision 17`` makes both logs print ``deltaT`` and the Courant numbers to the last
bit; the NeoN solver prints them with ``repr``.

**Time steps.** The deltaT sequences are equal bit for bit, on every executor and with
default as well as 1e-14 solver tolerances, also over 120 steps to ``t = 0.3``. The
Courant numbers alone do not explain that: ``nfb.courant_numbers`` reproduces
CourantNo.H/alphaCourantNo.H on equal fields (the first CoNum values, before and after the
first step, agree to 1e-14, the round-off of the step-1 fields), and from the next step on
the fields, and with them CoNum, differ by the solver tolerance (2e-8 there, at most 1.3e-3
over the run). What makes deltaT exact is ``adjustableRunTime``: each step is snapped to
the time left to the next write divided by a whole number of steps, so a Courant number
off by 1e-3 picks the same step count and the same double. Hence the sequence is asserted
equal, the first three CoNum to round-off and all Courant numbers within ``_COURANT_RTOL``
(4x the largest difference measured).

**Fields** at the write times are compared by relative L2 norm as in
``test_damBreak_comparison``; the ceilings are 3x the largest value measured over the five
write times (the same on Serial, CPU and GPU).
"""

import re
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest

from neofoam.io import DictFile

from ..incompressibleVoF.comparison_helpers import SubjectSolver, run_dambreak_comparison
from .test_damBreak_comparison import _set_common_atmosphere

_END_TIME = 0.1
_WRITE_INTERVAL = 0.025

# Measured max relL2 over t = 0.025..0.1: alpha.water 2.2e-6, p_rgh 6.5e-6, U 1.1e-3 (x3).
_CEILINGS = {"alpha.water": 6.5e-6, "p_rgh": 2e-5, "U": 3.3e-3}

# Measured: CoNum 1.3e-3, alphaCoNum 1.2e-4 at most (relative, over the 26 steps).
_COURANT_RTOL = 5e-3
# The first CoNum values agree to 1e-14, the round-off of the step-1 fields (x10).
_ROUND_OFF_RTOL = 1e-13
# The initial CoNum (0), the first step's (0) and the second's: round-off apart.
_ROUND_OFF_STEPS = 3

_DELTA_T = re.compile(r"^deltaT = (\S+)$", re.M)
_COURANT = re.compile(r"^Courant Number (?:mean: \S+ )?max: (\S+)$", re.M)
_INTERFACE_COURANT = re.compile(r"^Interface Courant Number (?:mean: \S+ )?max: (\S+)$", re.M)


def _logged(pattern: re.Pattern[str], log: Path) -> np.ndarray:
    return np.array([float(value) for value in pattern.findall(log.read_text())])


def _assert_time_steps_match(neon_case: Path, native_case: Path) -> None:
    """deltaT bit for bit, both Courant numbers within the solver-tolerance drift."""
    neon_log = neon_case / "log.incompressibleVoFNeoN"
    native_log = native_case / "log.interFoam"
    delta_t = _logged(_DELTA_T, neon_log)
    np.testing.assert_array_equal(
        delta_t, _logged(_DELTA_T, native_log), err_msg=f"deltaT sequence, {neon_case.name}"
    )
    assert delta_t.size >= 20, f"only {delta_t.size} time steps logged in {neon_log}"
    np.testing.assert_allclose(
        _logged(_COURANT, neon_log)[:_ROUND_OFF_STEPS],
        _logged(_COURANT, native_log)[:_ROUND_OFF_STEPS],
        rtol=_ROUND_OFF_RTOL,
        atol=0,
        err_msg=f"CoNum before the solver tolerance enters, {neon_case.name}",
    )
    for pattern in (_COURANT, _INTERFACE_COURANT):
        np.testing.assert_allclose(
            _logged(pattern, neon_log),
            _logged(pattern, native_log),
            rtol=_COURANT_RTOL,
            atol=0,
            err_msg=f"{pattern.pattern}, {neon_case.name}",
        )


def _prepare(executor: str) -> Callable[[Path], None]:
    """commonBCs, sigma 0, the NeoN executor, diagonal p_rgh, full-precision logs."""

    def prepare(case: Path) -> None:
        _set_common_atmosphere(case)
        transport = DictFile(case / "constant" / "transportProperties")
        transport.set("sigma", 0)
        transport.write()
        control = DictFile(case / "system" / "controlDict")
        control.set("executor", executor)
        control.set("writePrecision", 17)
        control.write()
        solution = DictFile(case / "system" / "fvSolution")
        for name in ("p_rgh", "p_rghFinal"):
            solution.set(("solvers", name, "preconditioner"), "diagonal")
        solution.write()

    return prepare


def _subject(
    executor: str, inspect_cases: Optional[Callable[[Path, Path], None]] = None
) -> SubjectSolver:
    """The NeoN solver on ``executor``, with the field ceilings and optional log checks."""
    return SubjectSolver(
        name="incompressibleVoFNeoN",
        command=[sys.executable, "-m", "neofoam.cli.app", "solver", "incompressiblevofneon"],
        prepare_case=_prepare(executor),
        rel_l2_ceilings=_CEILINGS,
        inspect_cases=inspect_cases,
    )


@pytest.mark.parametrize("executor", ["Serial", "CPU", "GPU"])
def test_adaptive_damBreak_time_steps_match_interFoam(executor: str) -> None:
    run_dambreak_comparison(
        tutorial_name="damBreak",
        native_solver="interFoam",
        case_prefix=f"damBreak_neon_adaptive_{executor}_dt",
        times=(_END_TIME, _WRITE_INTERVAL),
        subject=_subject(executor, inspect_cases=_assert_time_steps_match),
    )


@pytest.mark.parametrize("executor", ["Serial", "CPU", "GPU"])
def test_adaptive_damBreak_fields_match_interFoam_within_the_solver_tolerance(
    executor: str,
) -> None:
    run_dambreak_comparison(
        tutorial_name="damBreak",
        native_solver="interFoam",
        case_prefix=f"damBreak_neon_adaptive_{executor}",
        times=(_END_TIME, _WRITE_INTERVAL),
        subject=_subject(executor),
    )
