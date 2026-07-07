# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
DummyModel2 for DummySolver.

Demonstrates a minimal model: single field, single operation using a config.
"""

from typing import Any
from pathlib import Path
from pydantic import Field

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import InitStep
from neofoam.io import BaseConfig, IOStrategy, YAML

from .dummy_model import DummyModelInterface, Model


@IOStrategy(YAML("model2_config.yaml"))
class Model2Config(BaseConfig):
    step_size: float = Field(gt=0, description="Step increment per call")


model2 = Model("DummyModel2").register_with(DummyModelInterface)
model2.config(Model2Config)


@model2.load
def load(case_dir: Path, instance_id: str) -> Model2Config:
    return Model2Config.load(case_dir=case_dir, validate=False)


@model2.detect
def detect_model() -> bool:
    return True


@model2.build
def build() -> list[InitStep]:
    """BUILD stage: create a single field."""
    return [
        InitStep(
            name="model_field3",
            initializer=lambda _ctx: 500.0,
            depends_on=["domain"],
            category="fields",
        ),
    ]


@model2.operation(operation_number="2.8", depends_on=["solver_step2"])
def model2_step1(
    self: Any,
    model_field3: float,
    cfg: Model2Config,
) -> FieldUpdates:
    if not hasattr(self, "_step1_count"):
        self._step1_count = 0
    self._step1_count += 1
    return FieldUpdates({"model_field3": model_field3 + cfg.step_size})
