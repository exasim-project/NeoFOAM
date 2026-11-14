# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoFNeoN against native interFoam on ``tutorials/damBreak``, per executor.

Both cases are the tutorial (blockMesh + setFields) with ``sigma 0`` (the NeoN solver has
no surface tension) and a fixed ``deltaT 1e-3`` (the adaptive step is
``test_damBreak_adaptive_time_step``'s), so both take the same 50 steps to ``t = 0.05``,
compared every 0.01. The NeoN solver runs in a subprocess per executor: Kokkos and
``Foam::Time`` initialise once per process.

Both solvers use a diagonal instead of the tutorial's DIC ``p_rgh`` preconditioner. NeoN
maps DIC to Ginkgo's ``Ic``. Its ParIc factorisation takes the square root of the
diagonal, which is negative in the ``laplacian(rAUf, p_rgh)`` matrix, so it produces NaN.
That shows up as a floating-point trap on the host, an illegal memory access on CUDA, and
~640 CG iterations per solve when FPE trapping is off (OpenFOAM's DIC-PCG needs ~20).
Both also solve the MULESCorr predictor with diagonal PBiCGStab instead of the tutorial's
``smoothSolver``, which NeoN maps to Ginkgo's Ir and OpenFOAM runs as symGaussSeidel.

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

**Common boundary conditions** (``commonBCs``). The same run with the atmosphere set to
patch types both codes evaluate identically, applied to both cases: ``p_rgh`` fixedValue 0
(what NeoN's context-free ``totalPressure`` already is, and the tutorial's value wherever
no air enters) and ``U`` zeroGradient (what NeoN maps ``pressureInletOutletVelocity`` to,
and that patch's outflow branch); ``alpha.water`` keeps inletOutlet, which NeoN evaluates
with ``phi`` (``test/algorithms/vof/test_inlet_outlet_parity.py``), and the walls keep noSlip,
fixedFluxPressure and zeroGradient. So only interFoam's side changes. With all solver
tolerances at 1e-14 the U difference at ``t = 0.05`` falls to 1.1e-4 (tutorial BCs: 7e-4),
alpha.water to 2e-8 and p_rgh to 5e-7. What is left of U is not tolerance: the first step
agrees to 6e-13, and from the second step on the difference sits in the cells under the
atmosphere patch (nu = 0 leaves it unchanged; an upwind div(rhoPhi,U) shrinks it 5x).
At the default tolerances below, alpha.water differs as much as with the tutorial BCs, while
the largest p_rgh and U differences are 3.6x and 2.8x smaller, so their ceilings (again 3x
the largest relL2 measured) are tighter than the tutorial ones.
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
#   alpha.water 1.95e-6, p_rgh 4.21e-6, U 7.19e-4 (x3 = ceiling).
_CEILINGS = {"alpha.water": 5.9e-6, "p_rgh": 1.3e-5, "U": 2.2e-3}

# commonBCs, measured the same way: alpha.water 1.95e-6, p_rgh 1.18e-6, U 2.58e-4 (x3).
_COMMON_BC_CEILINGS = {"alpha.water": 5.9e-6, "p_rgh": 3.6e-6, "U": 7.8e-4}

# The atmosphere patch types NeoN and OpenFOAM evaluate identically (see the docstring).
_COMMON_ATMOSPHERE = {
    "p_rgh": {"type": "fixedValue", "value": "uniform 0"},
    "U": {"type": "zeroGradient"},
}


def _set_common_atmosphere(case: Path) -> None:
    for field_name, patch in _COMMON_ATMOSPHERE.items():
        field_file = DictFile(case / "0" / field_name)
        field_file.set(("boundaryField", "atmosphere"), patch)
        field_file.write()


def _prepare(executor: str, common_bcs: bool) -> Callable[[Path], None]:
    """sigma 0, fixed deltaT, the NeoN executor (interFoam ignores it), common matrix solvers."""

    def prepare(case: Path) -> None:
        if common_bcs:
            _set_common_atmosphere(case)
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
        solution.set(("solvers", '"alpha.water.*"', "solver"), "PBiCGStab")
        solution.set(("solvers", '"alpha.water.*"', "preconditioner"), "diagonal")
        solution.write()

    return prepare


@pytest.mark.parametrize(
    ("common_bcs", "ceilings"),
    [(False, _CEILINGS), (True, _COMMON_BC_CEILINGS)],
    ids=["tutorialBCs", "commonBCs"],
)
@pytest.mark.parametrize("executor", ["Serial", "CPU", "GPU"])
def test_damBreak_matches_interFoam_within_the_solver_tolerance(
    executor: str, common_bcs: bool, ceilings: dict[str, float]
) -> None:
    run_dambreak_comparison(
        tutorial_name="damBreak",
        native_solver="interFoam",
        case_prefix=f"damBreak_neon_{executor}{'_commonBCs' if common_bcs else ''}",
        times=(_END_TIME, _WRITE_INTERVAL),
        subject=SubjectSolver(
            name="incompressibleVoFNeoN",
            command=[sys.executable, "-m", "neofoam.cli.app", "solver", "incompressiblevofneon"],
            prepare_case=_prepare(executor, common_bcs),
            rel_l2_ceilings=ceilings,
        ),
    )
