# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
MultiModel (model4) for DummySolver.

Demonstrates multiple instances of the same spec discovered from a YAML config:
- detect returns instance IDs from config keys
- each instance gets its own config and uniquely-named field
"""

from typing import Any
from pathlib import Path

import yaml
from pydantic import Field

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import InitStep
from neofoam.framework.model import ModelRuntime
from neofoam.io import BaseConfig

from .dummy_model import DummyModelInterface, Model


model4 = Model("MultiModel").register_with(DummyModelInterface)

CONFIG_FILE = "model4_config.yaml"


@model4.config
class Model4Config(BaseConfig):
    scale: float = Field(gt=0, description="Multiplicative scale factor")
    offset: float = Field(description="Additive offset")


@model4.detect
def detect_model(case_dir: Path) -> list[str]:
    config_path = case_dir / CONFIG_FILE
    if not config_path.exists():
        return []
    with open(config_path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return []
    return list(data.keys())


@model4.load
def load(case_dir: Path, entry: dict[str, Any]) -> Model4Config:
    """Fallback load for detect path: reads config from YAML file."""
    instance_id = entry["name"]
    config_path = case_dir / CONFIG_FILE
    with open(config_path) as f:
        data = yaml.safe_load(f)
    fields = data[instance_id]
    return Model4Config.model_construct(**fields)


@model4.build
def build(self: ModelRuntime, config: Model4Config) -> list[InitStep]:
    field_name = (
        f"model_field4_{self.name}" if self.name != self.spec.name else "model_field4"
    )
    return [
        InitStep(
            name=field_name,
            initializer=lambda _ctx: 0.0,
            depends_on=["domain"],
            category="fields",
        ),
    ]


@model4.operation(operation_number="2.95", depends_on=["solver_step2"])
def model4_step1(
    self: Any,
    ctx: Context,
    cfg: Model4Config,
) -> FieldUpdates:
    field_name = (
        f"model_field4_{self.name}" if self.name != self.spec.name else "model_field4"
    )
    val = ctx.fields[field_name]
    return FieldUpdates({field_name: val * cfg.scale + cfg.offset})
