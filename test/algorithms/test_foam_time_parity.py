# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity: FoamTime must match the real pybFoam.Time step-for-step.

Builds an actual ``pybFoam.Time`` from a temporary case and drives both clocks
with the identical ``while loop():``, asserting equality on the shared
advancement quantities (``value``/``deltaTValue``/``outputTime``/``timeName``).
Field IO is *not* compared — that is the one place behaviour is meant to differ.

Skips cleanly when pybFoam or the OpenFOAM environment needed to construct a
``Time`` is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neofoam.algorithms.foam_time import FoamTime

pybFoam = pytest.importorskip("pybFoam")

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
    try:
        return pybFoam.Time(str(tmp_path), "case")
    except Exception as exc:  # noqa: BLE001 — env-dependent construction
        pytest.skip(f"cannot construct pybFoam.Time (OpenFOAM env?): {exc}")


@pytest.mark.parametrize(
    ("write_control", "write_interval"),
    [("timeStep", 2), ("runTime", 0.2), ("adjustableRunTime", 0.2)],
)
def test_parity_with_pybfoam_time(
    tmp_path: Path, write_control: str, write_interval: float
) -> None:
    params = dict(
        end_time=0.5,
        delta_t=0.1,
        write_control=write_control,
        write_interval=write_interval,
    )
    rt = _make_case(tmp_path, **params)
    foam = FoamTime(start_time=0.0, **params)  # type: ignore[arg-type]

    step = 0
    while True:
        running_ref = rt.loop()
        running_py = foam.loop()
        assert running_ref == running_py, f"loop() diverged at step {step}"
        if not running_ref:
            break
        step += 1
        assert rt.value() == pytest.approx(foam.value()), f"value at step {step}"
        assert rt.deltaTValue() == pytest.approx(foam.deltaTValue()), (
            f"deltaT at step {step}"
        )
        assert rt.outputTime() == foam.outputTime(), f"outputTime at step {step}"
        assert rt.timeName() == foam.timeName(), f"timeName at step {step}"

    assert step >= 5  # the case actually ran
