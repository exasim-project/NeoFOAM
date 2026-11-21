# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``SolverSpec.instantiate``: ``argv[0]`` is a program name; ``on_instantiate`` is called first."""

import importlib
from typing import Any

import pytest

from neofoam.framework.solver import Solver

SPEC_MODULE: Any = importlib.import_module("neofoam.framework.solver.spec")


def test_instantiate_empty_argv_gets_a_program_name() -> None:
    assert Solver("emptyArgv").instantiate().argv == ["neofoam"]
    assert Solver("emptyArgv").instantiate(argv=[]).argv == ["neofoam"]


def test_instantiate_keeps_a_given_argv() -> None:
    argv = ["prog", "-case", "x"]

    assert Solver("givenArgv").instantiate(argv=argv).argv == argv


def test_instantiate_calls_on_instantiate_with_the_argv_before_the_runtime_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = Solver("onInstantiate")
    events: list[Any] = []
    spec.on_instantiate = lambda argv: events.append(("on_instantiate", argv))
    runtime = SPEC_MODULE.SolverRuntime

    def recorded_runtime(**kwargs: Any) -> Any:
        events.append("runtime")
        return runtime(**kwargs)

    monkeypatch.setattr(SPEC_MODULE, "SolverRuntime", recorded_runtime)

    spec.instantiate(argv=["prog", "-case", "x"])
    spec.instantiate()

    assert events == [
        ("on_instantiate", ["prog", "-case", "x"]),
        "runtime",
        ("on_instantiate", ["neofoam"]),
        "runtime",
    ]
