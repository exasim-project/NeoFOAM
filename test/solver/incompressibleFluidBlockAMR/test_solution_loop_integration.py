# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-9 / C7 — the reused framework solutionLoop drives the engine to endTime.

The pure-Python time loop advances the blockamr engine; write steps emit AMReX
plotfiles.
"""

import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR import run  # noqa: E402


def test_loop_reaches_endtime_and_writes_plotfiles(blockamr_session, box_case):
    ctx = run(["incompressibleFluidBlockAMR"])

    # controlDict: endTime 0.05, deltaT 0.01 -> 5 steps reached.
    assert ctx.time.index >= 5
    assert ctx.time.value == pytest.approx(0.05, abs=1e-9)

    # writeInterval 2 -> at least one plotfile directory written.
    plotfiles = sorted(box_case.glob("plt*"))
    assert len(plotfiles) >= 1
    assert all(p.is_dir() for p in plotfiles)
