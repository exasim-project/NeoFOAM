# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Phase-2 regression tests for multi-config registration on a SolverSpec.

Locks in:
- ``spec.config(Cls)`` callable multiple times.
- Single registration → ``runtime.config`` is the instance itself.
- Multi registration → ``runtime.config`` is a ``SimpleNamespace`` keyed
  by snake-case class name; missing instances are silently omitted.
- Type-injection by ``find_config_by_type`` walks the namespace and
  picks each config by its annotation.
- Same-class registered twice is a no-op (not an error).
"""

from types import SimpleNamespace
from typing import Annotated, Any, Optional

import pytest

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


class ConfigA(BaseConfig):
    a: int = 1


class ConfigB(BaseConfig):
    b: str = "default"


def _build_runner(*instances: Any) -> StagedInitRunner:
    init = StagedInitSpec.build("MultiCfgInit")

    @init.load
    def _load() -> LoadResult:
        return LoadResult(core_models=list(instances), optional_models=[])

    @init.build
    def _build(_c: list[Any], _o: list[Any]) -> list[Any]:
        return []

    return StagedInitRunner(init.finalize())


def _attach_initializer(spec: Any, runner: StagedInitRunner) -> None:
    """Register the standard ``@spec.initializer`` that just runs *runner*."""

    @spec.initializer
    def _init(
        self: Any, init: Annotated[StagedInitRunner, Depends(lambda: runner)]
    ) -> Context:
        return init.run()


def test_single_registration_yields_instance() -> None:
    spec = Solver("S1")
    spec.config(ConfigA)

    cfg_a = ConfigA(a=5)
    runner = _build_runner(cfg_a)
    _attach_initializer(spec, runner)

    runtime = spec.instantiate()
    runtime.initialize()
    assert runtime.config is cfg_a


def test_multi_registration_yields_namespace() -> None:
    spec = Solver("S2")
    spec.config(ConfigA)
    spec.config(ConfigB)

    cfg_a, cfg_b = ConfigA(a=3), ConfigB(b="hi")
    runner = _build_runner(cfg_a, cfg_b)
    _attach_initializer(spec, runner)

    runtime = spec.instantiate()
    runtime.initialize()

    assert isinstance(runtime.config, SimpleNamespace)
    assert runtime.config.config_a is cfg_a
    assert runtime.config.config_b is cfg_b


def test_multi_registration_with_missing_instance() -> None:
    """If only one matching instance is in core_models, the namespace still works."""
    spec = Solver("S3")
    spec.config(ConfigA)
    spec.config(ConfigB)

    cfg_a = ConfigA(a=11)
    runner = _build_runner(cfg_a)  # no ConfigB instance
    _attach_initializer(spec, runner)

    runtime = spec.instantiate()
    runtime.initialize()

    assert isinstance(runtime.config, SimpleNamespace)
    assert runtime.config.config_a is cfg_a
    assert not hasattr(runtime.config, "config_b")


def test_type_injection_walks_namespace() -> None:
    """Each BaseConfig-annotated param finds its instance by type from the namespace."""
    spec = Solver("S4")
    spec.config(ConfigA)
    spec.config(ConfigB)

    cfg_a, cfg_b = ConfigA(a=7), ConfigB(b="x")
    runner = _build_runner(cfg_a, cfg_b)

    captured: dict[str, Any] = {}
    _attach_initializer(spec, runner)

    @spec.execution_graph_step
    def _graph(
        self: Any,
        a: ConfigA,
        b: ConfigB,
        domain_name: Optional[str] = None,
    ) -> tuple[StepBuilder, Operations]:
        captured["a"] = a
        captured["b"] = b
        return StepBuilder(), Operations()

    runtime = spec.instantiate()
    runtime.initialize()
    runtime.execution_graph()

    assert captured["a"] is cfg_a
    assert captured["b"] is cfg_b


def test_duplicate_registration_is_idempotent() -> None:
    """Registering the same class twice is a no-op, not an error."""
    spec = Solver("S5")
    spec.config(ConfigA)
    spec.config(ConfigA)
    assert spec._config_classes == [ConfigA]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
