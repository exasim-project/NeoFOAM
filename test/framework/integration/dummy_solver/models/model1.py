# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Generic model1 for DummySolver.

Demonstrates Model API with 3-stage initialization pattern and new config features.
"""

from typing import Any
from pydantic import Field

from foamadapter.framework.context import FieldUpdates
from foamadapter.framework.initialization.lazy_init import LazyInit
from foamadapter.framework.initialization import ConfigContext

from .dummy_model import DummyModelInterface, Model
from ..dummy_init import BaseConfig
from foamadapter.io import IOStrategy, YAML


@IOStrategy(
    YAML("model1_config.yaml", required=False, description="Model1 configuration")
)
class Model1Config(BaseConfig):
    """Configuration for Model1 (generic physics model)."""

    enabled: bool = True
    prop1: float = Field(ge=0, le=1, description="Property 1 must be between 0 and 1")
    prop2: float = Field(gt=0, description="Property 2 must be positive")
    parameters: dict[str, Any] = Field(default_factory=dict)


@IOStrategy(
    YAML(
        "model1_step_config.yaml",
        required=False,
        description="Model1 step configuration",
    )
)
class Model1StepConfig(BaseConfig):
    """Operation-specific configuration for Model1 steps."""

    factor: float = 0.01
    use_absolute: bool = True


# NEW: Create model - NO with_config needed, configs auto-discovered from operations
model1 = Model("DummyModel1").register_with(DummyModelInterface)

# NO @model1.load needed - auto-generated from discovered configs


@model1.detect
def detect_model() -> bool:
    """
    Detection: Check if model should be enabled.

    For now, always return True. In production, could check for
    specific configuration files or case setup.
    """
    return True


@model1.resolve
def resolve(config: ConfigContext) -> None:
    """RESOLVE stage: Connect to other models if needed."""
    pass


@model1.build
def build() -> list[LazyInit]:
    """
    BUILD stage: Create LazyInit objects for fields.

    Returns list of LazyInit objects for fields managed by this model.
    """

    def create_mf1() -> dict[str, Any]:
        """Create model field 1."""
        return {"name": "model_field1", "value": 300.0, "units": "mu1"}

    def create_mf2() -> dict[str, Any]:
        """Create model field 2 - uses config from load stage."""
        # Access Model1Config from the loaded configs dict
        main_config = (
            model1.config()
            if len(model1._configs) == 1
            else model1._configs.get("main") or list(model1._configs.values())[0]
        )
        # Find Model1Config instance
        for cfg in model1._configs.values():
            if isinstance(cfg, Model1Config):
                return {"name": "model_field2", "value": cfg.prop2, "units": "mu2"}
        # Fallback: if no Model1Config found, use default value
        return {"name": "model_field2", "value": 1000.0, "units": "mu2"}

    return [
        LazyInit(
            name="model_field1",
            initializer=create_mf1,
            depends_on=["domain"],
        ),
        LazyInit(
            name="model_field2",
            initializer=create_mf2,
            depends_on=["domain"],
        ),
    ]


@model1.operation(
    operation_number="2.5",
    depends_on=["solver_step1"],
)
def model1_step1(
    self: Any, field1: float, model_field1: float, step_config: Model1StepConfig
) -> FieldUpdates:
    """
    First model step with operation-specific config.

    Inserted between solver step 1 and 2.
    Uses step_config for operation-specific parameters (auto-discovered and injected).
    """
    self._step1_count += 1

    # Generic update using operation-specific config
    value = abs(field1) if step_config.use_absolute else field1
    mf1_new = model_field1 + value * step_config.factor

    return FieldUpdates({"model_field1": mf1_new})


@model1.operation(
    operation_number="2.7",
    depends_on=["model1_step1"],
)
def model1_step2(self: Any, main: Model1Config) -> FieldUpdates:
    """
    Second model step - uses main config (auto-discovered and injected).

    Demonstrates auto-injection from type annotations.
    """
    self._step2_count += 1

    # Generic update using model config - calculate from config values
    mf2_new = main.prop2 / main.prop1  # Use config from main

    return FieldUpdates({"model_field2": mf2_new})
