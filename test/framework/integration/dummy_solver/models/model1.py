# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Generic model1 for DummySolver.

Demonstrates Model API with 3-stage initialization pattern.
"""

from typing import Any
from pathlib import Path


from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.initialization.lazy_init import LazyInit

from .dummy_model import Model
from ..dummy_init import Model1Config


model1 = Model("DummyModel1")


# Model state (runtime counters)
model1._step1_count = 0
model1._step2_count = 0


@model1.load
def load_config() -> Model1Config:
    """
    LOAD stage: Load configuration from YAML file.

    Returns:
        Configuration object for this model
    """
    config_path = Path(__file__).parent.parent / "configs" / "model1_config.yaml"
    return Model1Config.load(config_path)


@model1.detect
def detect_model() -> bool:
    """
    Detection: Check if model should be enabled.

    For now, always return True. In production, could check for
    specific configuration files or case setup.
    """
    return True


# Convenience property access
@property
def prop1(self) -> float:
    return self._load_result.prop1 if self._load_result else 0.0


@property
def prop2(self) -> float:
    return self._load_result.prop2 if self._load_result else 0.0


# Bind properties to model instance
type(model1).prop1 = prop1
type(model1).prop2 = prop2


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
        config = model1._load_result
        return {"name": "model_field2", "value": config.prop2, "units": "mu2"}

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
def model1_step1(self: Any, ctx: Context) -> FieldUpdates:
    """
    First model step.

    Inserted between solver step 1 and 2.
    """
    self._step1_count += 1

    f1 = ctx.fields["field1"]  # Returns float directly
    mf1 = ctx.fields["model_field1"]  # Returns float directly

    # Generic update
    mf1_new = mf1 + abs(f1) * 0.01

    return FieldUpdates({"model_field1": mf1_new})


@model1.operation(operation_number="2.7", depends_on=["model1_step1"])
def model1_step2(self: Any, ctx: Context) -> FieldUpdates:
    """
    Second model step - uses config from load stage.
    """
    self._step2_count += 1

    # Generic update using model config
    param1 = ctx.models["config"]["param1"]
    config = self._load_result
    mf2_new = param1 / config.prop1  # Use config from load stage

    return FieldUpdates({"model_field2": mf2_new})
