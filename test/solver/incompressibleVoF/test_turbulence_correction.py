# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""When the VoF solver's ``turbulence_correction`` step actually corrects.

interFoam and interIsoFoam guard the call with ``if (pimple.turbCorr())``, so
under ``nOuterCorrectors > 1`` the turbulence — and with it ``nut`` — is updated
**once per time step**, on the final outer corrector. Correcting in every outer
corrector feeds the next iteration's momentum assembly an eddy viscosity native
has not produced yet, which moves ``p_rgh``, then ``phi``, then alpha.

The step is driven here exactly as the solver's inner loop drives it: a real
:class:`PimpleControl` built by the production factory from a real case dict
(``cases/pimple_outer3`` / ``cases/pimple_turb_every_outer``, which differ in
the single ``turbOnFinalIterOnly`` key), stepped through one full outer loop.
The turbulence model itself is a counter — what is under test is the schedule,
not what ``correct()`` computes, and a live two-phase turbulence model would
need a whole solver run to build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neofoam.solver.incompressibleVoF.incompressibleVoF import turbulence_correction
from neofoam.solver.incompressibleVoF.models.pressure_velocity.control_factory import (
    create_pimple_control,
)

_CASES = Path(__file__).parent / "cases"


class _CountingTurbulence:
    """Records how many times the solver asked for a correction."""

    def __init__(self) -> None:
        self.corrections = 0

    def correct(self) -> None:
        self.corrections += 1


def _corrections_per_outer_iteration() -> list[int]:
    """Cumulative correction count after each outer corrector of one time step.

    Reads the PIMPLE dict of the *current working directory*, as the solver does.
    """
    control = create_pimple_control({})
    turbulence = _CountingTurbulence()
    counts = []
    while control.loop():
        turbulence_correction(turbulence, control)
        counts.append(turbulence.corrections)
    return counts


@pytest.mark.parametrize(
    "case_name, expected_cumulative_corrections",
    [
        # nOuterCorrectors 3, turbOnFinalIterOnly at its native default.
        pytest.param("pimple_outer3", [0, 0, 1], id="turbOnFinalIterOnly_default"),
        # The same case with turbOnFinalIterOnly no.
        pytest.param("pimple_turb_every_outer", [1, 2, 3], id="turbOnFinalIterOnly_no"),
    ],
)
def test_the_turbulence_is_corrected_on_the_outer_iterations_the_case_asks_for(
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    expected_cumulative_corrections: list[int],
) -> None:
    monkeypatch.chdir(_CASES / case_name)
    assert _corrections_per_outer_iteration() == expected_cumulative_corrections


def test_a_case_without_a_turbulence_model_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Laminar cases carry no turbulence model at all; the gate must not be the
    # thing that decides whether the step is safe to run.
    monkeypatch.chdir(_CASES / "pimple_outer3")
    control = create_pimple_control({})
    while control.loop():
        assert turbulence_correction(None, control) == {}
