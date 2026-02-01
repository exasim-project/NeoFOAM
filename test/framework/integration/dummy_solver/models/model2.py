# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Generic model2 for DummySolver.

Demonstrates simplified model with 3-stage initialization.
"""

from typing import Any
from foamadapter.framework.context import Context, FieldUpdates
from foamadapter.framework.initialization.lazy_init import LazyInit

from .dummy_model import Model

model2 = Model("DummyModel2")

# Model state
model2._step1_count = 0


@model2.detect
def detect_model() -> bool:
    """Always enabled for testing purposes."""
    return True


@model2.build
def build() -> list[LazyInit]:
    """BUILD stage: Create LazyInit objects for fields."""

    def create_mf3() -> dict[str, Any]:
        return {"name": "model_field3", "value": 500.0, "units": "mu3"}

    return [
        LazyInit(
            name="model_field3",
            initializer=create_mf3,
            depends_on=["domain"],
        ),
    ]


@model2.operation(operation_number="2.8", depends_on=["solver_step2"])
def model2_step1(
    self: Any,
    ctx: Context,
) -> FieldUpdates:
    """Model 2 step."""
    self._step1_count += 1
    mf3 = ctx.fields["model_field3"]
    return FieldUpdates({"model_field3": mf3 + 1.0})


# Export
auxiliary = model2
AuxiliaryModel = model2
