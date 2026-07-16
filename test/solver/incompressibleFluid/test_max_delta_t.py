# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the maxDeltaT optional model and its timeStepConstraint contribution."""

# NOTE: no `from __future__ import annotations` — keep annotations live.

from pathlib import Path
from typing import Any, cast

import pytest

from pydantic import ValidationError

from neofoam.algorithms.solution_loop.interfaces import VGREAT, timeStepConstraint
from neofoam.tooling.casebuild import from_template, patch
from neofoam.framework.context import Context
from neofoam.framework.model import BoundModelInterface, ModelRuntime
from neofoam.solver.incompressibleFluid.models.incompressibleFluidModel import (
    incompressibleFluidModel,
)
from neofoam.solver.incompressibleFluid.models.max_delta_t import (
    MaxDeltaTConfig,
    maxDeltaT,
)

_CASES = Path(__file__).parent / "cases"
_BASE = _CASES / "controldict_base"


def _max_runtime(cap: float = 0.5) -> ModelRuntime:
    return ModelRuntime(
        spec=maxDeltaT, name="maxDeltaT", config=MaxDeltaTConfig(maxDeltaT=cap)
    )


def test_model_is_registered_in_the_family_catalog() -> None:
    names = {spec.name for spec in incompressibleFluidModel.all_specs()}
    assert "maxDeltaT" in names


def test_registering_the_model_twice_keeps_one_catalog_entry() -> None:
    maxDeltaT.register_with(incompressibleFluidModel)
    names = [spec.name for spec in incompressibleFluidModel.all_specs()]
    assert names.count("maxDeltaT") == 1


def test_model_owns_the_control_dict_config() -> None:
    assert maxDeltaT._config_class is MaxDeltaTConfig
    assert cast(Any, MaxDeltaTConfig).io_config.file == "system/controlDict"


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_config_rejects_non_positive_cap(bad: float) -> None:
    with pytest.raises(ValidationError):
        MaxDeltaTConfig(maxDeltaT=bad)


def test_contribution_caps_delta_t_when_model_active() -> None:
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(timeStepConstraint, [_max_runtime(0.5)], ctx)
    assert bound() == pytest.approx(0.5)


def test_contribution_excluded_when_model_inactive() -> None:
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(timeStepConstraint, [], ctx)
    assert bound() == VGREAT


def test_fold_raises_when_config_is_absent() -> None:
    # The model is active but carries no config -> the fold raises, naming the
    # interface, the contribution, and the unresolved parameter.
    rt = ModelRuntime(spec=maxDeltaT, name="maxDeltaT", config=None)
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(timeStepConstraint, [rt], ctx)
    with pytest.raises(ValueError) as exc:
        bound()
    message = str(exc.value)
    assert "timeStepConstraint" in message
    assert "max_delta_t_limit" in message
    assert "cfg" in message


def test_config_presence_activates_contribution_through_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = (
        from_template(_BASE)
        | patch("system/controlDict", {"adjustTimeStep": True, "maxDeltaT": 0.5})
    ).build_at(tmp_path / "case")
    monkeypatch.chdir(case.path)

    detected = {rt.name: rt for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "maxDeltaT" in detected
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(timeStepConstraint, [detected["maxDeltaT"]], ctx)
    assert bound() == pytest.approx(0.5)


def test_absent_config_leaves_contribution_unfolded_through_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = (
        from_template(_BASE) | patch("system/controlDict", {"adjustTimeStep": True})
    ).build_at(tmp_path / "case")
    monkeypatch.chdir(case.path)

    detected = {rt.name: rt for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "maxDeltaT" not in detected
    ctx = Context(fields={}, models={})
    bound = BoundModelInterface(timeStepConstraint, list(detected.values()), ctx)
    assert bound() == VGREAT


def test_two_runs_in_one_process_flip_participation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case1 = (
        from_template(_BASE)
        | patch("system/controlDict", {"adjustTimeStep": True, "maxDeltaT": 0.5})
    ).build_at(tmp_path / "case1")
    monkeypatch.chdir(case1.path)

    m1 = {rt.name: rt for rt in incompressibleFluidModel.detect_models(Path("."))}
    b1 = BoundModelInterface(
        timeStepConstraint, [m1["maxDeltaT"]], Context(fields={}, models={})
    )
    assert b1() == pytest.approx(0.5)

    case2 = (
        from_template(_BASE) | patch("system/controlDict", {"adjustTimeStep": True})
    ).build_at(tmp_path / "case2")
    monkeypatch.chdir(case2.path)

    m2 = {rt.name: rt for rt in incompressibleFluidModel.detect_models(Path("."))}
    b2 = BoundModelInterface(
        timeStepConstraint, list(m2.values()), Context(fields={}, models={})
    )
    assert b2() == VGREAT


def test_model_inactive_when_no_control_dict_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert maxDeltaT.run_detect() is False


def test_max_delta_t_inactive_when_adjust_time_step_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # maxDeltaT present but adjustTimeStep no: OpenFOAM keeps a fixed step, so the
    # contribution must stay inactive (the symmetric guard to courant's no-adjust arm).
    case = (
        from_template(_BASE)
        | patch("system/controlDict", {"adjustTimeStep": False, "maxDeltaT": 0.5})
    ).build_at(tmp_path / "case")
    monkeypatch.chdir(case.path)

    detected = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "maxDeltaT" not in detected


def test_max_delta_t_inactive_when_adjust_time_step_key_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The adjustTimeStep key omitted entirely short-circuits detection -> inactive.
    # The base controlDict already omits adjustTimeStep, so patching maxDeltaT alone
    # reproduces the "key absent" case without any line removal.
    case = (
        from_template(_BASE) | patch("system/controlDict", {"maxDeltaT": 0.5})
    ).build_at(tmp_path / "case")
    monkeypatch.chdir(case.path)

    detected = {rt.name for rt in incompressibleFluidModel.detect_models(Path("."))}
    assert "maxDeltaT" not in detected
