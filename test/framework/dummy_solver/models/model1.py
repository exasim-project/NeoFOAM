# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Generic model1 for DummySolver.

Demonstrates Model API with configuration loaded from YAML.
"""

from typing import Any, Annotated
from pathlib import Path

from pydantic import BaseModel

from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.initialization import Depends
from foamadapter.framework.initialization.lazy_init import LazyInit

from .dummy_model import Model


model1 = Model("DummyModel1")


from ..dummy_init import Model1Config


# Model state (runtime counters)
model1._step1_count = 0
model1._step2_count = 0
model1._config: Model1Config | None = None


def get_config() -> Model1Config:
    """
    Load configuration from configs folder.
    """
    if model1._config is None:
        # Try to load from YAML, fall back to defaults
        config_path = Path(__file__).parent.parent / "configs" / "model1_config.yaml"
        model1._config = Model1Config.load(config_path)

    return model1._config


# Convenience property access (for backward compatibility with tests)
@property
def prop1(self) -> float:
    return get_config().prop1


@property
def prop2(self) -> float:
    return get_config().prop2


# Bind properties to model instance
type(model1).prop1 = prop1
type(model1).prop2 = prop2


@model1.build_step
def build(config: Annotated[Model1Config, Depends(get_config)]) -> list[LazyInit]:
    """
    Build model fields using loaded configuration.

    Configuration is injected via Depends mechanism.

    Returns list of LazyInit objects for fields.
    """

    def create_mf1() -> dict[str, Any]:
        """Create model field 1."""
        return {"name": "model_field1", "value": 300.0, "units": "mu1"}

    def create_mf2() -> dict[str, Any]:
        """Create model field 2 - uses config."""
        cfg = get_config()
        return {"name": "model_field2", "value": cfg.prop2, "units": "mu2"}

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


@model1.configure_algorithm_step
def configure_algorithm(algorithm: Any) -> None:
    """Configure algorithm to include model1 effects."""
    if hasattr(algorithm, "_use_model1"):
        algorithm._use_model1 = True


@model1.operation(operation_number="2.5", depends_on=["solver_step1"])
def model1_step1(
    self: Any, ctx: Context, config: Annotated[Model1Config, Depends(get_config)]
) -> FieldUpdates:
    """
    First model step - configuration injected via Depends.

    Inserted between solver step 1 and 2.
    """
    self._step1_count += 1

    f1 = ctx.fields["field1"]  # Returns float directly
    mf1 = ctx.fields["model_field1"]  # Returns float directly

    # Generic update
    mf1_new = mf1 + abs(f1) * 0.01

    return FieldUpdates({"model_field1": mf1_new})


@model1.operation(operation_number="2.7", depends_on=["model1_step1"])
def model1_step2(
    self: Any, ctx: Context, config: Annotated[Model1Config, Depends(get_config)]
) -> FieldUpdates:
    """
    Second model step - uses injected config.
    """
    self._step2_count += 1

    mf2 = ctx.fields["model_field2"]  # Returns float directly

    # Generic update using model config
    param1 = ctx.models["config"]["param1"]
    mf2_new = param1 / config.prop1  # Use injected config

    return FieldUpdates({"model_field2": mf2_new})


# Export names
thermal = model1
ThermalModel = model1
