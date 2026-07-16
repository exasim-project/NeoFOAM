# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for create_fields helpers — the optional-model ctx.models projection."""

from pathlib import Path

import pytest

from neofoam.framework.model import BoundModelInterface
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


def test_solution_loop_step_binds_owned_interface_onto_owner_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The live @build auto-wiring: create_init emits a step that binds the
    # solutionLoop owner runtime's declared interfaces to the case's active
    # contributors and registers it under ctx.models["solutionLoop"]. Drive the
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
    bound = owner_runtime.bound_interfaces["timeStepConstraint"]
    assert isinstance(bound, BoundModelInterface)
    # loopCondition is owned + bound by the same step.
    condition = owner_runtime.bound_interfaces["loopCondition"]
    assert isinstance(condition, BoundModelInterface)

    # The owner is bound against an EMPTY Context — the GC-safe no-capture contract:
    # capturing the live pybFoam fields/models here would create a mesh-bound
    # reference cycle that segfaults at GC across in-process solver runs. The live
    # fold happens at CALL time with a Context passed by set_time_step (proven in
    # test_interface.py / test_courant.py). Pin "no live capture" with a unit
    # assertion so a regression to live-snapshot is caught here, not only by the
    # integration SIGBUS. (_ctx is white-box; the empty-Context contract is the point.)
    assert bound._ctx.fields == {}
    assert bound._ctx.models == {}
    assert condition._ctx.fields == {}
    assert condition._ctx.models == {}
