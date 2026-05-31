# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
DummyModel1 for DummySolver.

Demonstrates ModelSpec API with 3-stage initialization and auto config injection.
"""

from typing import Any, Callable
from pathlib import Path
from pydantic import Field

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import ConfigContext, InitStep
from neofoam.io import BaseConfig, IOStrategy, YAML

from .dummy_model import DummyModelInterface, Model


@IOStrategy(YAML("model1_config.yaml"))
class Model1Config(BaseConfig):
    """Configuration for Model1 (generic physics model)."""

    enabled: bool = True
    prop1: float = Field(ge=0, le=1, description="Property 1 must be between 0 and 1")
    prop2: float = Field(gt=0, description="Property 2 must be positive")
    parameters: dict[str, Any] = Field(default_factory=dict)


@IOStrategy(YAML("model1_step_config.yaml"))
class Model1StepConfig(BaseConfig):
    """Operation-specific configuration for Model1 steps."""

    factor: float = 0.01
    use_absolute: bool = True


# ModelSpec — immutable definition, registered once at module import
model1 = Model("DummyModel1").register_with(DummyModelInterface)

model1.config(Model1Config)
model1.config(Model1StepConfig)


@model1.load
def load(case_dir: Path, instance_id: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        main=Model1Config.load(case_dir=case_dir, validate=False),
        step_config=Model1StepConfig.load(case_dir=case_dir, validate=False),
    )


@model1.detect
def detect_model() -> bool:
    return True


@model1.resolve
def resolve(config: Any, ctx: ConfigContext) -> Any:
    """RESOLVE stage: no cross-model wiring needed for model1."""
    return config


def _make_field2_init(cfg: Model1Config) -> Callable[[dict[str, Any]], float]:
    def _init(_ctx: dict[str, Any]) -> float:
        return cfg.prop2

    return _init


@model1.build
def build(config: Any) -> list[InitStep]:
    """
    BUILD stage: create LazyInit objects for fields managed by this model.

    ``config`` is a SimpleNamespace with .main (Model1Config) and
    .step_config (Model1StepConfig) since two config types are discovered.
    """
    main = config.main if hasattr(config, "main") else config

    return [
        InitStep(
            name="model_field1",
            initializer=lambda _ctx: 300.0,
            depends_on=["domain"],
            category="fields",
        ),
        InitStep(
            name="model_field2",
            initializer=_make_field2_init(main),
            depends_on=["domain"],
            category="fields",
        ),
    ]


@model1.operation(
    operation_number="2.5",
    depends_on=["solver_step1"],
)
def model1_step1(
    self: Any,
    field1: float,
    model_field1: float,
    step_config: Model1StepConfig,
) -> FieldUpdates:
    """First model step — uses step_config (auto-injected)."""
    if not hasattr(self, "_step1_count"):
        self._step1_count = 0
    self._step1_count += 1
    value = abs(field1) if step_config.use_absolute else field1
    return FieldUpdates({"model_field1": model_field1 + value * step_config.factor})


@model1.operation(
    operation_number="2.7",
    depends_on=["model1_step1"],
)
def model1_step2(self: Any, main: Model1Config) -> FieldUpdates:
    """Second model step — uses main config (auto-injected)."""
    if not hasattr(self, "_step2_count"):
        self._step2_count = 0
    self._step2_count += 1
    return FieldUpdates({"model_field2": main.prop2 / main.prop1})
