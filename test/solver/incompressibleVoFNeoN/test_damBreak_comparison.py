# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoFNeoN against native interFoam on ``tutorials/damBreak``, per executor.

Both cases are the tutorial (blockMesh + setFields) with ``sigma 0`` (the NeoN solver has
no surface tension) and a fixed ``deltaT 1e-3`` (it has no adaptive time step), so both
take the same 50 steps to ``t = 0.05``, compared every 0.01. The NeoN solver runs in a
subprocess per executor: Kokkos and ``Foam::Time`` initialise once per process.

Both solvers use a diagonal instead of the tutorial's DIC ``p_rgh`` preconditioner. NeoN
maps DIC to Ginkgo's ``Ic``. Its ParIc factorisation takes the square root of the
diagonal, which is negative in the ``laplacian(rAUf, p_rgh)`` matrix, so it produces NaN.
That shows up as a floating-point trap on the host, an illegal memory access on CUDA, and
~640 CG iterations per solve when FPE trapping is off (OpenFOAM's DIC-PCG needs ~20).

The fields cannot agree to round-off, for two documented reasons:

* **Linear-solver tolerance.** ``p_rgh`` is solved to ``relTol 0.05`` (``p_rghFinal``
  ``relTol 0``, ``tolerance 1e-7``) and the MULESCorr predictor to ``1e-8`` by Ginkgo in
  NeoN and by OpenFOAM's own solvers, which stop at different iterates. Tightening all
  tolerances to 1e-14 in both solvers cuts the alpha.water and p_rgh differences at
  ``t = 0.05`` by 20x and 50x (to 6e-7 and 4e-6), so these are solver-tolerance gaps.
* **Boundary conditions.** The ``atmosphere`` patch is NeoN's: ``totalPressure`` on
  ``p_rgh`` is a fixedValue at ``p0`` (OpenFOAM subtracts ``0.5*rho*|U|^2`` on inflow
  faces) and ``pressureInletOutletVelocity`` on ``U`` is zeroGradient (OpenFOAM fixes the
  tangential inflow component to zero). With tight tolerances ``U`` still differs by 7e-4,
  which is this gap: the air the falling column draws in through the patch.

The ceilings are the largest relL2 measured over the five write times, times 3 (margin
for Ginkgo-version and executor round-off; Jacobi-CG gives the same fields to 3 digits on
Serial, CPU and GPU).
"""

import sys
from pathlib import Path
from typing import Callable

import pytest

from neofoam.io import DictFile

from ..incompressibleVoF.comparison_helpers import SubjectSolver, run_dambreak_comparison

_END_TIME = 0.05
_WRITE_INTERVAL = 0.01
_DELTA_T = 1e-3

# Measured max relL2 over t = 0.01..0.05, the same on every executor:
#   alpha.water 5.0e-6, p_rgh 4.9e-6, U 1.1e-3 (x3 = ceiling).
_CEILINGS = {"alpha.water": 1.5e-5, "p_rgh": 1.5e-5, "U": 3.5e-3}


def _prepare(executor: str) -> Callable[[Path], None]:
    """sigma 0, fixed deltaT, the NeoN executor (interFoam ignores it), diagonal p_rgh."""

    def prepare(case: Path) -> None:
        transport = DictFile(case / "constant" / "transportProperties")
        transport.set("sigma", 0)
        transport.write()
        control = DictFile(case / "system" / "controlDict")
        control.set("adjustTimeStep", "no")
        control.set("deltaT", _DELTA_T)
        control.set("executor", executor)
        control.write()
        solution = DictFile(case / "system" / "fvSolution")
        for name in ("p_rgh", "p_rghFinal"):
            solution.set(("solvers", name, "preconditioner"), "diagonal")
        solution.write()

    return prepare


@pytest.mark.parametrize("executor", ["Serial", "CPU", "GPU"])
def test_damBreak_matches_interFoam_within_the_solver_tolerance(executor: str) -> None:
    run_dambreak_comparison(
        tutorial_name="damBreak",
        native_solver="interFoam",
        case_prefix=f"damBreak_neon_{executor}",
        times=(_END_TIME, _WRITE_INTERVAL),
        subject=SubjectSolver(
            name="incompressibleVoFNeoN",
            command=[sys.executable, "-m", "neofoam.cli.app", "solver", "incompressiblevofneon"],
            prepare_case=_prepare(executor),
            rel_l2_ceilings=_CEILINGS,
        ),
    )
