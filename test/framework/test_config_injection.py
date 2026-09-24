# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for ``config_injection``.

``_find_config_by_type`` is the lookup the operation-config injector uses to
pull a specific ``BaseConfig`` out of a runtime's ``config`` — whether that
config is the instance itself or one attribute of a ``SimpleNamespace``
holding several. These tests pin the direct-match, namespace-search, and
not-found (``ValueError``) paths.

``_create_runtime_config_wrapper`` is the operation wrapper built around that
lookup. A config-typed parameter has to coexist with the marker annotations
every other parameter uses (``Annotated[..., "models"]`` / ``"fields"``), so the
wrapper hands the non-config remainder to the shared ``DependencyResolver`` — the
turbulence closures declare both kinds in one signature.
"""

from types import SimpleNamespace
from typing import Annotated, Any

import pytest

from neofoam.framework.config_injection import (
    _create_runtime_config_wrapper,
    _discover_configs_from_signature,
    _find_config_by_type,
)
from neofoam.framework.context import Context, FieldUpdates
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


class GainConfig(BaseConfig):
    """A coefficients-style config injected into an operation by type."""

    gain: float = 3.0


def _gain_op(
    coeffs: GainConfig,
    helper: Annotated[Any, "models"],
    x: Annotated[Any, "fields"],
) -> FieldUpdates:
    """One config parameter next to a model marker and a field marker."""
    return FieldUpdates({"x": helper(x) * coeffs.gain})


def test_runtime_config_wrapper_injects_config_alongside_marker_parameters() -> None:
    runtime = SimpleNamespace(config=GainConfig())
    wrapper = _create_runtime_config_wrapper(
        _gain_op, _discover_configs_from_signature(_gain_op), runtime
    )
    ctx = Context(fields={"x": 2.0}, models={"helper": lambda v: v + 1.0}, mesh={})

    wrapper(ctx)

    assert ctx.fields["x"] == pytest.approx(9.0)  # (2 + 1) * 3
