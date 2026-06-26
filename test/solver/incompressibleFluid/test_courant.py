# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the CFL timeStepConstraint contribution (solver-side, pybFoam)."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

import importlib
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.algorithms.solution_loop.interfaces import VGREAT, timeStepConstraint
from neofoam.framework.context import Context
from neofoam.framework.model import BoundModelInterface, ModelRuntime
from neofoam.solver.incompressibleFluid.models.courant import (
    CourantConfig,
    courant,
    courant_limit,
)
from neofoam.solver.incompressibleFluid.models.incompressibleFluidModel import (
    incompressibleFluidModel,
)

courant_mod = importlib.import_module(
    "neofoam.solver.incompressibleFluid.models.courant"
)

_CASES = Path(__file__).parent / "cases"


def _courant_runtime(max_co: float = 1.0) -> ModelRuntime:
    return ModelRuntime(
        spec=courant, name="courant", config=CourantConfig(maxCo=max_co)
    )


def test_contribution_lives_under_the_solver_not_the_framework() -> None:
    assert courant_limit.__module__.startswith("neofoam.solver.incompressibleFluid")


def test_courant_contribution_limits_delta_t_like_the_cfl_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(courant_mod, "computeCFLNumber", lambda phi: [2.0])
    ctx = Context(fields={"phi": object(), "deltaT": 0.1}, models={})
    bound = BoundModelInterface(timeStepConstraint, [_courant_runtime()], ctx)
    assert bound() == pytest.approx(0.05)  # 0.1 * 1.0 / 2.0


def test_quiescent_flow_yields_no_opinion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(courant_mod, "computeCFLNumber", lambda phi: [0.0])
    ctx = Context(fields={"phi": object(), "deltaT": 0.1}, models={})
    bound = BoundModelInterface(timeStepConstraint, [_courant_runtime()], ctx)
    assert bound() == VGREAT


def test_contribution_excluded_when_model_inactive() -> None:
    ctx = Context(fields={"phi": object(), "deltaT": 0.1}, models={})
    bound = BoundModelInterface(timeStepConstraint, [], ctx)  # courant not active
    assert bound() == VGREAT


def test_model_is_registered_in_the_family_catalog() -> None:
    names = {spec.name for spec in incompressibleFluidModel.all_specs()}
    assert "courant" in names


def test_model_and_config_discoverable_without_a_case() -> None:
    # Registration != activation: no case dir needed to list the model + its config.
    names = {spec.name for spec in incompressibleFluidModel.all_specs()}
    assert "courant" in names
    assert courant._config_class is CourantConfig


def test_config_presence_drives_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_CASES / "maxCo_present")
    on = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "courant" in on
    assert "maxDeltaT" not in on

    monkeypatch.chdir(_CASES / "maxDeltaT_absent")
    off = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "courant" not in off

    # maxCo present but adjustTimeStep no: OpenFOAM keeps a fixed step, so courant
    # must stay inactive (preserves the hotRoom/cavity/cylinder2D fixed-step parity).
    monkeypatch.chdir(_CASES / "maxCo_no_adjust")
    fixed = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "courant" not in fixed

    # adjustTimeStep key omitted entirely: the short-circuit on a missing key -> inactive.
    monkeypatch.chdir(_CASES / "no_adjust_key")
    no_key = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "courant" not in no_key


def test_contribution_resolves_live_phi_and_config_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bound against an EMPTY ctx (no live capture); the live ctx is passed at call
    # time and is where phi + deltaT resolve, while cfg comes from the runtime.
    monkeypatch.setattr(courant_mod, "computeCFLNumber", lambda phi: [2.0])
    bound = BoundModelInterface(
        timeStepConstraint,
        [_courant_runtime(max_co=1.0)],
        Context(fields={}, models={}),
    )
    live = Context(fields={"phi": object(), "deltaT": 0.1}, models={})
    assert bound(live) == pytest.approx(0.05)  # 0.1 * 1.0 / 2.0
    with pytest.raises(ValueError, match="no provider supplies it"):
        bound()  # empty bound ctx -> phi/deltaT absent


def test_model_inactive_when_no_control_dict_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert courant.run_detect() is False
