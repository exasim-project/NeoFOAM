# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity: SolutionLoop advancement must match the real pybFoam.Time step-for-step.

Builds an actual ``pybFoam.Time`` from a temporary case and drives both clocks
with the identical ``while loop:``, asserting equality on the shared advancement
quantities (``value``/``deltaT``/``write_time``/``timeName``). Field IO is *not*
compared — that is the one place behaviour is meant to differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pybFoam
import pytest

from neofoam.algorithms.solution_loop.config import _WRITE_CONTROL_ALIASES
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoop

_CONTROL_DICT = """\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          {delta_t};
writeControl    {write_control};
writeInterval   {write_interval};
"""


def _make_case(tmp_path: Path, **kw: object) -> Any:
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir()
    (case / "system" / "controlDict").write_text(_CONTROL_DICT.format(**kw))
    return pybFoam.Time(str(tmp_path), "case")


@pytest.mark.parametrize(
    ("write_control", "write_interval"),
    [("timeStep", 2), ("runTime", 0.2), ("adjustableRunTime", 0.2)],
)
def test_parity_with_pybfoam_time(
    tmp_path: Path, write_control: str, write_interval: float
) -> None:
    rt = _make_case(
        tmp_path,
        end_time=0.5,
        delta_t=0.1,
        write_control=write_control,
        write_interval=write_interval,
    )
    loop = SolutionLoop(
        state=LoopState(
            value=0.0,
            delta_t=0.1,
            end_time=0.5,
            start_time=0.0,
            write_control=_WRITE_CONTROL_ALIASES.get(write_control, write_control),
            write_interval=write_interval,
        )
    )

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
