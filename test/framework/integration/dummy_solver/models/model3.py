# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
CoupledModel (model3) for DummySolver.

Demonstrates:
- Conditional operation dispatch (coupled vs standalone) driven by config.coupled.
- Nested sub-model (Accumulator) built into ctx.models and accessed via ctx.
"""

from typing import Any
from pathlib import Path

from pydantic import Field

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import ConfigContext, InitStep
from neofoam.framework.operations import Operation, Operations, SequentialOp
from neofoam.framework.types import OperationMetadata, OperationNumber
from neofoam.io import BaseConfig, IOStrategy, YAML

from .dummy_model import DummyModelInterface, Model


@IOStrategy(YAML("model3_config.yaml"))
class Model3Config(BaseConfig):
    """Configuration for CoupledModel."""

    mode: str = Field(description="'coupled' or 'standalone'")
    damping: float = Field(gt=0, le=1, description="Damping factor")
    coupled: bool = False


class Accumulator:
    """Nested sub-model that tracks a running sum."""

    def __init__(self, initial: float = 0.0):
        self.value = initial
        self.count = 0

    def accumulate(self, delta: float, damping: float) -> float:
        self.value += delta * damping
        self.count += 1
        return self.value


# ModelSpec — immutable definition
model3 = Model("CoupledModel").register_with(DummyModelInterface)
model3.config(Model3Config)


@model3.load
def load(case_dir: Path, instance_id: str) -> Model3Config:
    return Model3Config.load(case_dir=case_dir, validate=False)


@model3.detect
def detect_model() -> bool:
    return True


@model3.resolve
def resolve(config: Model3Config, ctx: ConfigContext) -> Model3Config:
    """Check if DummyModel1 is active and store as coupled flag in config."""
    from neofoam.framework.model import ModelRuntime

    coupled = any(
        isinstance(v, ModelRuntime) and v.spec.name == "DummyModel1"
        for v in ctx.all().values()
    )
    return config.model_copy(update={"coupled": coupled})


@model3.build
def build(config: Model3Config) -> list[InitStep]:
    """Create fields and the nested accumulator sub-model."""

    def create_model3_field(_ctx: dict[str, Any]) -> float:
        return 0.0

    def create_accumulator(_ctx: dict[str, Any]) -> Accumulator:
        return Accumulator(initial=0.0)

    return [
        InitStep(
            name="model3_field",
            initializer=create_model3_field,
            depends_on=["domain"],
            category="fields",
        ),
        InitStep(
            name="accumulator",
            initializer=create_accumulator,
            depends_on=[],
            category="models",
        ),
    ]


# ---------------------------------------------------------------------------
# Operation variants: coupled (uses model_field1) vs standalone (uses field1)
# ---------------------------------------------------------------------------


@model3.operation(operation_number="2.9", depends_on=["solver_step2"])
def coupled_step(
    ctx: Any,
    model3_field: float,
    model_field1: float,
    cfg: Model3Config,
) -> FieldUpdates:
    """Coupled variant: combines model3_field with model1's model_field1."""
    damped = ctx.models["accumulator"].accumulate(model_field1, cfg.damping)
    return FieldUpdates({"model3_field": model3_field + damped * 0.01})


@model3.operation(operation_number="2.9", depends_on=["solver_step2"])
def standalone_step(
    ctx: Any,
    model3_field: float,
    field1: float,
    cfg: Model3Config,
) -> FieldUpdates:
    """Standalone variant: uses solver's field1 directly."""
    damped = ctx.models["accumulator"].accumulate(field1, cfg.damping * 0.1)
    return FieldUpdates({"model3_field": model3_field + damped * 0.01})


# ---------------------------------------------------------------------------
# Conditional dispatch via @operation_collection
# ---------------------------------------------------------------------------


@model3.operation_collection
def collected_operations(self: Any) -> Operations:
    """
    Choose between coupled_step and standalone_step based on config.coupled.

    ``self`` is the ModelRuntime; ``self.config.coupled`` was set during RESOLVE.
    """
    from neofoam.framework.config_injection import (
        _discover_configs_from_signature,
        _create_runtime_config_wrapper,
    )

    raw_func, metadata = (
        self.spec._operations[0] if self.config.coupled else self.spec._operations[1]
    )

    discovered = _discover_configs_from_signature(raw_func)
    wrapped = _create_runtime_config_wrapper(raw_func, discovered, self)

    op = Operation(
        func=SequentialOp(wrapped),
        metadata=OperationMetadata(
            op_name="model3_step",
            operation_number=OperationNumber(metadata["operation_number"]),
            depends_on=metadata["depends_on"] or [],
            before=metadata["before"] or [],
        ),
    )
    ops = Operations()
    ops.add(op)
    return ops
