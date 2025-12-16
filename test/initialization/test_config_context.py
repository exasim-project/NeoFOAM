# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for ConfigContext."""

from foamadapter.framework.initialization import ConfigContext
from .initialization_test_models import MockTurbulenceModel, MockTransportModel


def test_config_context_register_and_get():
    """Test basic config context operations."""
    config = ConfigContext()
    model = MockTurbulenceModel()

    config.register("turb", model)

    assert config.get("turb") is model
    assert config.get("unknown") is None


def test_config_context_all():
    """Test getting all models from config context."""
    config = ConfigContext()
    turb = MockTurbulenceModel()
    trans = MockTransportModel()

    config.register("turbulence", turb)
    config.register("transport", trans)

    all_models = config.all()

    assert len(all_models) == 2
    assert all_models["turbulence"] is turb
    assert all_models["transport"] is trans


def test_config_context_contains():
    """Test checking if model is registered."""
    config = ConfigContext()
    model = MockTurbulenceModel()

    config.register("turb", model)

    assert config.contains("turb")
    assert not config.contains("unknown")


def test_config_context_multi_region():
    """Test multi-region support with dot notation."""
    config = ConfigContext(current_region="fluid")

    # Register in current region (fluid)
    fluid_model = MockTurbulenceModel()
    config.register("temperature", fluid_model)

    # Register in another region (solid)
    solid_model = MockTransportModel()
    config.register("temperature", solid_model, region="solid")

    # Same region access (no dot)
    assert config.get("temperature") is fluid_model

    # Cross-region access (with dot)
    assert config.get("solid.temperature") is solid_model
    assert config.get("fluid.temperature") is fluid_model

    # Contains checks
    assert config.contains("temperature")
    assert config.contains("solid.temperature")
    assert config.contains("fluid.temperature")
    assert not config.contains("solid.velocity")


def test_config_context_regions():
    """Test regions property."""
    config = ConfigContext(current_region="fluid")
    config.register("temp", MockTurbulenceModel())
    config.register("temp", MockTransportModel(), region="solid")

    regions = config.regions
    assert "fluid" in regions
    assert "solid" in regions
    assert len(regions) == 2
