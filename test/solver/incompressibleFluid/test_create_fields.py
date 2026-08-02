# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for create_fields helpers — the optional-model ctx.models projection."""

from pathlib import Path

import pytest

from neofoam.solver.incompressibleFluid.create_fields import (
    _optional_models_by_name,
    create_init,
)

_CASES = Path(__file__).parent


class _FakeRuntime:
    """Stands in for a ModelRuntime: the projection only reads ``.name``."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_optional_models_are_keyed_by_their_model_name() -> None:
    # The projection keys each detected runtime under its model name, which is
    # exactly what the interface owner-gate (owner.name in ctx.models) reads.
    cap = _FakeRuntime("maxDeltaT")
    cfl = _FakeRuntime("courant")
    assert _optional_models_by_name([cap, cfl]) == {"maxDeltaT": cap, "courant": cfl}


def test_no_optional_models_yields_an_empty_mapping() -> None:
    assert _optional_models_by_name([]) == {}


def test_solution_loop_step_registers_the_owner_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # create_init emits a step that registers the solutionLoop owner runtime
    # under ctx.models["solutionLoop"]. The runtime carries no case-bound state:
    # its gather hooks (timeStepConstraint / loopCondition) are bound to the live
    # Context per injection (proven in test_interface.py / test_courant.py), so
    # nothing mesh-bound is captured across in-process solver runs. Drive the
    # real graph (read dicts only — no lazy step is executed, so no mesh/pybFoam
    # object is created) and run exactly that one step.
    case = _CASES / "val_pitzDaily"
    # detect_and_create reads dictionaries by relative path, so run from the case.
    monkeypatch.chdir(case)
    runner = create_init(case_dir=case)
    runner.run_load()
    steps = runner.run_build()

    wire_step = next(s for s in steps if s.name == "models.solutionLoop")
    assert wire_step.depends_on == ["models.solution_loop"]

    owner_runtime = wire_step.initializer({})
    assert owner_runtime.spec.name == "solutionLoop"
    assert {"timeStepConstraint", "loopCondition"} <= set(owner_runtime.spec.declared_interfaces)
