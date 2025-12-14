# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for ModelRegistry."""

from foamadapter.framework.initialization import ModelRegistry
from .test_fixtures import TestTurbulenceModel, TestTransportModel


def test_registry_register_and_get():
    """Test basic registry operations."""
    registry = ModelRegistry()
    model = TestTurbulenceModel()

    registry.register("turb", model)

    assert registry.get("turb") is model
    assert registry.get("unknown") is None


def test_registry_all():
    """Test getting all models from registry."""
    registry = ModelRegistry()
    turb = TestTurbulenceModel()
    trans = TestTransportModel()

    registry.register("turbulence", turb)
    registry.register("transport", trans)

    all_models = registry.all()

    assert len(all_models) == 2
    assert all_models["turbulence"] is turb
    assert all_models["transport"] is trans


def test_registry_contains():
    """Test checking if model is registered."""
    registry = ModelRegistry()
    model = TestTurbulenceModel()

    registry.register("turb", model)

    assert registry.contains("turb")
    assert not registry.contains("unknown")
