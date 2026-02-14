# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Test that models are properly registered with DummyModelInterface.
"""

from .models import model1, model2
from .models.dummy_model import DummyModelInterface
from neofoam.core.plugin_system import PluginSystem


def test_models_registered_with_dummy_interface():
    """Test that model1 and model2 are registered with DummyModelInterface."""
    # Import models to trigger registration

    # Get the registry
    registry = PluginSystem.get_registered("DummyModelInterface")
    assert registry is not None

    # Check that both models are in the registry
    plugin_names = registry.get_plugin_names()
    assert "DummyModel1" in plugin_names
    assert "DummyModel2" in plugin_names
    assert len(plugin_names) == 2


def test_create_models_accessible_via_plugin_system():
    """Test that models can be accessed through the plugin system."""

    # Create instances through the create method - now returns ModelInstance directly
    m1 = DummyModelInterface.create(config={"model_type": "DummyModel1"})
    assert isinstance(m1, type(model1))  # Both are ModelInstance objects
    assert m1.name == "DummyModel1"

    m2 = DummyModelInterface.create(config={"model_type": "DummyModel2"})
    assert isinstance(m2, type(model2))
    assert m2.name == "DummyModel2"
