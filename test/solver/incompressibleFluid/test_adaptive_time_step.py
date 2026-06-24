# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the adaptiveTimeStep optional time-step model.

CFL is no longer baked into the core loop: this opt-in model owns the
``adjustTimeStep``/``maxCo``/``maxDeltaT`` keys of ``controlDict``, installs its
deltaT constraints into the built engine, and registers the Courant measurement
the loop body publishes. Pure-Python apart from ``detect`` (reads a dict via
pybFoam) and the config IO — no OpenFOAM case run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.solution_loop import SolutionLoop
from neofoam.solver.incompressibleFluid.models.adaptive_time_step import (
    CourantControlConfig,
    adaptiveTimeStep,
    detect_model,
)

_CONTROLDICT = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
application     pimpleFoam;
endTime         1;
deltaT          0.1;
{adjust}
"""


def _write_controldict(case: Path, adjust: str) -> None:
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(_CONTROLDICT.format(adjust=adjust))


def _state() -> LoopState:
    return LoopState(value=0.0, delta_t=0.1, end_time=1.0)


# -- the model is a flagged optional toggle --------------------------------


def test_model_is_a_registered_toggle() -> None:
    assert adaptiveTimeStep.name == "adaptiveTimeStep"
    assert adaptiveTimeStep.toggle is True
    assert adaptiveTimeStep.toggle_label == "Adaptive time step (Courant)"


# -- config validation -----------------------------------------------------


def test_config_requires_maxco() -> None:
    with pytest.raises(Exception):
        CourantControlConfig()  # maxCo is required
    with pytest.raises(Exception):
        CourantControlConfig(maxCo=0.0)  # gt=0
    cfg = CourantControlConfig(maxCo=0.5)
    assert cfg.maxCo == 0.5
    assert cfg.maxDeltaT is None
    assert cfg.adjustTimeStep is True


# -- detect reads controlDict ----------------------------------------------


def test_detect_true_when_adjust_time_step_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_controldict(tmp_path, "adjustTimeStep yes;\nmaxCo 1;")
    monkeypatch.chdir(tmp_path)
    assert detect_model() is True


def test_detect_false_when_adjust_time_step_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_controldict(tmp_path, "adjustTimeStep no;")
    monkeypatch.chdir(tmp_path)
    assert detect_model() is False


def test_detect_false_when_key_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_controldict(tmp_path, "")
    monkeypatch.chdir(tmp_path)
    assert detect_model() is False


# -- build installs constraints + the measurement provider -----------------


def test_build_installs_courant_and_maxdeltat() -> None:
    steps = adaptiveTimeStep._build_func(CourantControlConfig(maxCo=1.0, maxDeltaT=0.5))
    by_name = {s.name: s for s in steps}
    assert set(by_name) == {
        "models.adaptive_time_step",
        "models.measurement_provider.courant",
    }

    loop = SolutionLoop(state=_state())
    install = by_name["models.adaptive_time_step"]
    assert "models.solution_loop" in install.depends_on
    install.initializer({"models.solution_loop": loop})
    assert {type(c).__name__ for c in loop.constraints} == {
        "CourantConstraint",
        "MaxDeltaTConstraint",
    }


def test_build_without_maxdeltat_installs_only_courant() -> None:
    steps = adaptiveTimeStep._build_func(CourantControlConfig(maxCo=1.0))
    by_name = {s.name: s for s in steps}
    loop = SolutionLoop(state=_state())
    by_name["models.adaptive_time_step"].initializer({"models.solution_loop": loop})
    assert {type(c).__name__ for c in loop.constraints} == {"CourantConstraint"}


def test_provider_is_registered_and_callable() -> None:
    steps = adaptiveTimeStep._build_func(CourantControlConfig(maxCo=1.0))
    by_name = {s.name: s for s in steps}
    provider = by_name["models.measurement_provider.courant"].initializer({})
    assert callable(provider)


def test_constraints_cfl_limit_the_step() -> None:
    # end-to-end on the pure-Python engine: install -> publish -> adjust
    steps = adaptiveTimeStep._build_func(CourantControlConfig(maxCo=1.0))
    by_name = {s.name: s for s in steps}
    loop = SolutionLoop(state=_state())
    by_name["models.adaptive_time_step"].initializer({"models.solution_loop": loop})
    loop.publish("courant", 2.0)  # too fast
    loop.adjust_delta_t()
    assert loop.state.delta_t == 0.05  # 0.1 * 1.0 / 2.0


def test_detect_signature_takes_no_args() -> None:
    # detect is a no-arg predicate (reads the cwd case), like boussinesq's
    import inspect

    sig = inspect.signature(detect_model)
    assert list(sig.parameters) == []
