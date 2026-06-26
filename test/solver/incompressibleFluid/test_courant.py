# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the CFL timeStepConstraint contribution (solver-side, pybFoam)."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import pytest

# The module imports computeCFLNumber/surfaceScalarField from pybFoam at import time.
pytest.importorskip("pybFoam")

from neofoam.algorithms.solution_loop.interfaces import timeStepConstraint
from neofoam.framework.context import Context
from neofoam.solver.incompressibleFluid.models import courant as courant_mod
from neofoam.solver.incompressibleFluid.models.courant import (
    CourantConfig,
    courant_limit,
)


def test_contribution_lives_under_the_solver_not_the_framework() -> None:
    assert courant_limit.__module__.startswith("neofoam.solver.incompressibleFluid")


def test_courant_contribution_limits_delta_t_like_the_cfl_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the pybFoam CFL number so the test needs no mesh; the contribution's job
    # is the arithmetic deltaT * maxCo / Co, matching the legacy CourantConstraint.
    monkeypatch.setattr(courant_mod, "computeCFLNumber", lambda phi: [2.0])
    ctx = Context(
        fields={
            "phi": object(),  # sentinel: computeCFLNumber is stubbed
            "deltaT": 0.1,
            "cfg": CourantConfig(maxCo=1.0),
        },
        models={},
    )
    timeStepConstraint.activate(courant_limit)
    try:
        # 0.1 * 1.0 / 2.0 -> the CFL rule shrinks the step to 0.05
        assert timeStepConstraint.collect(ctx) == pytest.approx(0.05)
    finally:
        timeStepConstraint.deactivate(courant_limit)


def test_quiescent_flow_yields_no_opinion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(courant_mod, "computeCFLNumber", lambda phi: [0.0])
    ctx = Context(
        fields={"phi": object(), "deltaT": 0.1, "cfg": CourantConfig(maxCo=1.0)},
        models={},
    )
    timeStepConstraint.activate(courant_limit)
    try:
        from neofoam.algorithms.solution_loop.interfaces import VGREAT

        assert timeStepConstraint.collect(ctx) == VGREAT
    finally:
        timeStepConstraint.deactivate(courant_limit)
