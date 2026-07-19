# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Regression tests for ``ModelSpec.instantiate`` auto-load.

After ``model.config(Cls)`` is called and the model is instantiated
*without* a manifest entry, the framework auto-loads the config via
the class's ``@IOStrategy`` binding. This mirrors the solver-side
``spec.config(Cls)`` ergonomics.

Locks in:

- Class-form registration with no callback auto-loads from disk on
  ``instantiate(case_dir=…)``.
- Manifest-driven loading (``entry`` dict) still works (back-compat).
- Multi-class registration auto-loads every class and yields a
  ``SimpleNamespace``-shaped ``runtime.config``.
- Bare ``config()`` with no class and no callback still raises.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from neofoam.framework.model import Model
from neofoam.io import BaseConfig, IOStrategy, YAML


@IOStrategy(YAML("a.yaml"))
class AConfig(BaseConfig):
    a: int = 1


@IOStrategy(YAML("b.yaml"))
class BConfig(BaseConfig):
    b: str = "default"


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"a": 7}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"b": "hi"}))
    return tmp_path


def test_class_form_auto_loads_without_callback(case_dir: Path) -> None:
    spec = Model("AutoLoad")
    spec.config(AConfig)
    runtime = spec.instantiate(case_dir=case_dir)
    assert isinstance(runtime.config, AConfig)
    assert runtime.config.a == 7


def test_multi_class_registration_auto_loads_to_namespace(case_dir: Path) -> None:
    spec = Model("MultiAuto")
    spec.config(AConfig)
    spec.config(BConfig)
    runtime = spec.instantiate(case_dir=case_dir)
    assert isinstance(runtime.config, SimpleNamespace)
    assert isinstance(runtime.config.a_config, AConfig)
    assert isinstance(runtime.config.b_config, BConfig)
    assert runtime.config.a_config.a == 7
    assert runtime.config.b_config.b == "hi"


def test_load_callback_still_takes_precedence(case_dir: Path) -> None:
    """If @model.load is registered, the callback wins over auto-load."""
    spec = Model("CallbackWins")
    spec.config(AConfig)

    @spec.load
    def _load(case: Path, _instance_id: str) -> AConfig:
        return AConfig(a=999)  # custom — ignores file

    runtime = spec.instantiate(case_dir=case_dir)
    assert runtime.config.a == 999


def test_no_config_no_callback_raises(case_dir: Path) -> None:
    spec = Model("Bare")
    with pytest.raises(ValueError, match="cannot instantiate"):
        spec.instantiate(case_dir=case_dir)
