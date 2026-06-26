# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the CFL timeStepConstraint contribution (solver-side, pybFoam)."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import importlib
from pathlib import Path

import pytest

# The module imports computeCFLNumber/surfaceScalarField from pybFoam at import time.
pytest.importorskip("pybFoam")

from neofoam.algorithms.solution_loop.interfaces import VGREAT, timeStepConstraint
from neofoam.framework.context import Context
from neofoam.solver.incompressibleFluid.models.courant import (
    CourantConfig,
    courant,
    courant_limit,
)
from neofoam.solver.incompressibleFluid.models.incompressibleFluidModel import (
    incompressibleFluidModel,
)

# The package re-exports the `courant` model object under the same name as its
# submodule, so reach the module explicitly to monkeypatch its pybFoam helper.
courant_mod = importlib.import_module(
    "neofoam.solver.incompressibleFluid.models.courant"
)

_CASES = Path(__file__).parent / "cases"


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
        models={"courant": object()},  # the courant model is active for this case
    )
    # 0.1 * 1.0 / 2.0 -> the CFL rule shrinks the step to 0.05
    assert timeStepConstraint.collect(ctx) == pytest.approx(0.05)


def test_quiescent_flow_yields_no_opinion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(courant_mod, "computeCFLNumber", lambda phi: [0.0])
    ctx = Context(
        fields={"phi": object(), "deltaT": 0.1, "cfg": CourantConfig(maxCo=1.0)},
        models={"courant": object()},
    )
    assert timeStepConstraint.collect(ctx) == VGREAT


def test_contribution_excluded_when_model_inactive() -> None:
    # The courant model is not active for this case (absent from ctx.models), so its
    # contribution does not fold and the constraint stays fixed-step (VGREAT).
    ctx = Context(
        fields={"phi": object(), "deltaT": 0.1, "cfg": CourantConfig(maxCo=1.0)},
        models={},
    )
    assert timeStepConstraint.collect(ctx) == VGREAT


def test_model_is_registered_in_the_family_catalog() -> None:
    names = {spec.name for spec in incompressibleFluidModel.all_specs()}
    assert "courant" in names


def test_config_presence_drives_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    # maxCo present -> the courant model is detected active for the case.
    monkeypatch.chdir(_CASES / "maxCo_present")
    on = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "courant" in on

    # maxCo absent -> the courant model is not detected, independent of maxDeltaT.
    monkeypatch.chdir(_CASES / "maxDeltaT_absent")
    off = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "courant" not in off

    monkeypatch.chdir(_CASES / "maxCo_present")
    on = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "courant" in on
    assert "maxDeltaT" not in on  # maxCo activates ONLY courant, independently


def test_model_inactive_when_no_control_dict_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No system/controlDict in the working dir: the isfile guard degrades the
    # detect to inactive instead of raising.
    monkeypatch.chdir(tmp_path)
    assert courant.run_detect() is False
