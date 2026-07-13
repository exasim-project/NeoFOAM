# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Phase-1 regression tests for SolverSpec.config(...) wiring.

Locks in:
- SolverRuntime gains a ``.config`` field.
- ``SolverSpec.config(Cls)`` registers the class on the spec.
- After ``runtime.initialize()`` runs, the framework finds the matching
  instance from ``runtime.state.core_models`` and assigns it to
  ``runtime.config``.
- ``BaseConfig``-annotated parameters on ``@spec.execution_graph_step``
  are type-injected from ``runtime.config``.
"""

from typing import Annotated, Any, Optional


from neofoam.framework.context import Context
from neofoam.framework.initialization import Depends
from neofoam.framework.initialization.staged.runner import StagedInitRunner
from neofoam.framework.initialization.staged.spec import (
    LoadResult,
    StagedInitSpec,
)
from neofoam.framework.operations import Operations, StepBuilder
from neofoam.framework.solver import Solver
from neofoam.io import BaseConfig


class DummySolverConfig(BaseConfig):
    max_iter: int = 10


def test_solver_runtime_config_defaults_to_none_and_is_assignable() -> None:
    """A bare SolverRuntime carries ``config`` (None until populated) and holds it."""
    runtime = Solver("Bare").instantiate()
    assert runtime.config is None
    runtime.config = DummySolverConfig(max_iter=3)
    assert runtime.config.max_iter == 3


def test_solver_spec_config_registers_class() -> None:
    """Calling spec.config(Cls) stores the class on the spec."""
    spec = Solver("RegisterOnly")
    spec.config(DummySolverConfig)
    assert spec._config_class is DummySolverConfig


def _build_runner(solver_cfg: DummySolverConfig) -> StagedInitRunner:
    init = StagedInitSpec.build("DummyInit")

    @init.load
    def _load() -> LoadResult:
        return LoadResult(core_models=[solver_cfg], optional_models=[])

    @init.build
    def _build(_c: list[Any], _o: list[Any]) -> list[Any]:
        return []

    return StagedInitRunner(init.finalize())


def test_runtime_config_populated_from_core_models() -> None:
    """After initialize(), the registered config class instance is on runtime.config."""
    spec = Solver("ConfigPopulated")
    spec.config(DummySolverConfig)

    cfg = DummySolverConfig(max_iter=42)
    runner = _build_runner(cfg)

    @spec.initializer
    def _init(
        self: Any, init: Annotated[StagedInitRunner, Depends(lambda: runner)]
    ) -> Context:
        return init.run()

    runtime = spec.instantiate()
    runtime.initialize()

    assert runtime.config is cfg
    assert runtime.config.max_iter == 42


def test_execution_graph_step_receives_config_by_type() -> None:
    """A BaseConfig-typed param on @execution_graph_step is injected from runtime.config."""
    spec = Solver("InjectedGraph")
    spec.config(DummySolverConfig)

    cfg = DummySolverConfig(max_iter=7)
    runner = _build_runner(cfg)

    @spec.initializer
    def _init(
        self: Any, init: Annotated[StagedInitRunner, Depends(lambda: runner)]
    ) -> Context:
        return init.run()

    captured: dict[str, Any] = {}

    @spec.execution_graph_step
    def _graph(
        self: Any,
        injected_cfg: DummySolverConfig,
        domain_name: Optional[str] = None,
    ) -> tuple[StepBuilder, Operations]:
        captured["cfg"] = injected_cfg
        return StepBuilder(), Operations()

    runtime = spec.instantiate()
    runtime.initialize()
    runtime.execution_graph()

    assert captured["cfg"] is cfg
    assert captured["cfg"].max_iter == 7


def test_runtime_config_stays_none_when_no_class_registered() -> None:
    """No registration → no auto-population (back-compat path)."""
    spec = Solver("NoConfig")
    cfg = DummySolverConfig(max_iter=1)
    runner = _build_runner(cfg)

    @spec.initializer
    def _init(
        self: Any, init: Annotated[StagedInitRunner, Depends(lambda: runner)]
    ) -> Context:
        return init.run()

    runtime = spec.instantiate()
    runtime.initialize()
    assert runtime.config is None
