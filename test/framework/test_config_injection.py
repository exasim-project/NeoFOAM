# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for ``config_injection._find_config_by_type``.

``_find_config_by_type`` is the lookup the operation-config injector uses to
pull a specific ``BaseConfig`` out of a runtime's ``config`` — whether that
config is the instance itself or one attribute of a ``SimpleNamespace``
holding several. These tests pin the direct-match, namespace-search, and
not-found (``ValueError``) paths.
"""

from types import SimpleNamespace

import pytest

from neofoam.framework.config_injection import _find_config_by_type
from neofoam.io import BaseConfig


def test_find_config_by_type_raises_on_missing_type() -> None:
    class MyConfig(BaseConfig):
        x: int = 1

    class OtherConfig(BaseConfig):
        y: int = 2

    cfg = MyConfig()
    with pytest.raises(ValueError, match="OtherConfig"):
        _find_config_by_type(cfg, OtherConfig)


def test_find_config_by_type_finds_direct_match() -> None:
    class MyConfig(BaseConfig):
        x: int = 5

    cfg = MyConfig()
    assert _find_config_by_type(cfg, MyConfig) is cfg


def test_find_config_by_type_searches_namespace() -> None:
    class StepConfig(BaseConfig):
        factor: float = 0.01

    class MainConfig(BaseConfig):
        prop: float = 1.0

    ns = SimpleNamespace(step=StepConfig(), main=MainConfig())
    assert isinstance(_find_config_by_type(ns, StepConfig), StepConfig)
    assert isinstance(_find_config_by_type(ns, MainConfig), MainConfig)


def test_find_config_by_type_raises_when_not_in_namespace() -> None:
    class Missing(BaseConfig):
        pass

    ns = SimpleNamespace()
    with pytest.raises(ValueError, match="Missing"):
        _find_config_by_type(ns, Missing)
