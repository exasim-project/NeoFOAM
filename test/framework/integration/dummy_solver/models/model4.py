# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
MultiModel (model4) for DummySolver.

Demonstrates multiple instances of the same spec with no instance state:
- no @init_runtime — purely functional
- single operation using Model4Config
"""

from typing import Any
from pathlib import Path
from pydantic import Field

from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import InitStep
from neofoam.io import BaseConfig, IOStrategy, YAML

from .dummy_model import DummyModelInterface, Model


@IOStrategy(YAML("model4_config.yaml"))
class Model4Config(BaseConfig):
    scale: float = Field(gt=0, description="Multiplicative scale factor")
    offset: float = Field(description="Additive offset")


model4 = Model("MultiModel").register_with(DummyModelInterface)
model4.config(Model4Config)


@model4.detect
def detect_model() -> bool:
    return True


@model4.load
def load(case_dir: Path, instance_id: str) -> Model4Config:
    return Model4Config.load(case_dir=case_dir, validate=False)


@model4.build
def build() -> list[InitStep]:
    return [
        InitStep(
            name="model_field4",
            initializer=lambda _ctx: 0.0,
            depends_on=["domain"],
            category="fields",
        ),
    ]


@model4.operation(operation_number="2.95", depends_on=["solver_step2"])
def model4_step1(
    self: Any,
    model_field4: float,
    cfg: Model4Config,
) -> FieldUpdates:
    return FieldUpdates({"model_field4": model_field4 * cfg.scale + cfg.offset})
