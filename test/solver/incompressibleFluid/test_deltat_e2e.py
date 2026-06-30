# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: the solver folds the model-owned timeStepConstraint to set deltaT."""

import os
import shutil
from pathlib import Path

import pytest

from neofoam.solver.incompressibleFluid import run

from .comparison_helpers import setup_case

_CONTROL = """\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end};
deltaT          {dt};
writeControl    timeStep;
writeInterval   100000;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable no;
{adjust}
"""


def _write_control(case: Path, *, dt: float, end: float, maxco: bool) -> None:
    # OpenFOAM (and the courant detect gate) only honour maxCo when adjustTimeStep
    # is on; the fixed-step reference therefore turns adjustTimeStep off entirely
    # (TimeControlConfig also requires a maxCo whenever adjustTimeStep is yes).
    adjust = (
        "adjustTimeStep  yes;\nmaxCo           0.2;" if maxco else "adjustTimeStep  no;"
    )
    (case / "system" / "controlDict").write_text(
        _CONTROL.format(dt=dt, end=end, adjust=adjust)
    )


def _run(case: Path) -> float:
    here = Path.cwd()
    os.chdir(case)
    try:
        ctx = run(["incompressibleFluid"])
    finally:
        os.chdir(here)
    return float(ctx.time.delta_t)


def test_solver_constrains_delta_t_through_the_interface_fold() -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    source = repo_root / "tutorials" / "pitzDaily"
    constrained = repo_root / "test_cases" / "deltaT_constrained"
    fixed = repo_root / "test_cases" / "deltaT_fixed"
    dt0 = 0.01
    try:
        # blockMesh + restore 0/ via the shared helper, then overwrite controlDict.
        setup_case(source, constrained, end_time=dt0 * 3, write_interval=dt0 * 3)
        setup_case(source, fixed, end_time=dt0 * 3, write_interval=dt0 * 3)
        _write_control(constrained, dt=dt0, end=dt0 * 3, maxco=True)
        _write_control(fixed, dt=dt0, end=dt0 * 3, maxco=False)

        # Two real solver runs in ONE process (also the cross-run no-SIGBUS check).
        dt_constrained = _run(constrained)  # courant active -> CFL-limited
        dt_fixed = _run(fixed)  # no maxCo -> VGREAT fold -> fixed step

        assert dt_fixed == pytest.approx(dt0)
        assert dt_constrained < dt0  # the interface fold shrank the step
    finally:
        for tc in (constrained, fixed):
            if tc.exists():
                shutil.rmtree(tc)
