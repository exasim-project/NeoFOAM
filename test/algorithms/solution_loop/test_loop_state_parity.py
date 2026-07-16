# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity: SolutionLoop advancement must match the real pybFoam.Time step-for-step.

Both clocks are built from the *same* checked-in ``system/controlDict`` (one case
per write-control regime under ``cases/``), copied into ``tmp_path`` so the source
files are never mutated. The pybFoam side loads the case directly; the Python side
loads the same controlDict through ``TimeControlConfig`` + ``make_loop_state`` — the
production path — so no case content is fabricated in the test. Both are then driven
with the identical ``while loop:`` and asserted equal on the shared advancement
quantities (``value``/``deltaT``/``write_time``/``timeName``). Field IO is *not*
compared — that is the one place behaviour is meant to differ.

Adding a write-control regime is a new case directory + a ``parametrize`` entry, not
new test code.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pybFoam
import pytest

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.algorithms.solution_loop.solution_loop import (
    SolutionLoop,
    make_loop_state,
    make_solution_loop,
)

_CASES = Path(__file__).parent / "cases"


@pytest.mark.parametrize("case_name", ["timeStep", "runTime", "adjustableRunTime"])
def test_parity_with_pybfoam_time(tmp_path: Path, case_name: str) -> None:
    case = tmp_path / case_name
    shutil.copytree(_CASES / case_name, case)

    rt = pybFoam.Time(str(tmp_path), case_name)
    config = TimeControlConfig.load(case_dir=case)
    loop: SolutionLoop = make_solution_loop(config, make_loop_state(config))

    step = 0
    while True:
        running_ref = rt.loop()  # advances pyf.Time if running
        running_py = loop.run()
        if running_py:
            loop.advance()
        assert running_ref == running_py, f"loop() diverged at step {step}"
        if not running_ref:
            break
        step += 1
        s = loop.state
        assert rt.value() == pytest.approx(s.value), f"value at step {step}"
        assert rt.deltaTValue() == pytest.approx(s.delta_t), f"deltaT at step {step}"
        assert rt.outputTime() == s.write_time, f"write_time at step {step}"
        assert rt.timeName() == loop.timeName(), f"timeName at step {step}"

    assert step >= 5  # the case actually ran
