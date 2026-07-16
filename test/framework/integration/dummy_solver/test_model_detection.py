# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Registry unit test: the plugin system discovers the optional-model family.

Verifies that ``@register_with(DummyModelInterface)`` makes all four dummy
models detectable — ``detect_specs()`` returns them as ModelSpec objects and
the PluginSystem registry exposes them by name — which is the discovery step
``SolverSpec.models(...)`` relies on before any case is loaded.
"""

from .models.dummy_model import DummyModelInterface
from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelSpec


def test_detect_specs_returns_model_spec_objects() -> None:
    """detect_specs() must return ModelSpec instances, one per registered model."""
    specs = DummyModelInterface.detect_specs()
    assert len(specs) == 4
    assert all(isinstance(s, ModelSpec) for s in specs)
    names = {s.name for s in specs}
    assert names == {"DummyModel1", "DummyModel2", "CoupledModel", "MultiModel"}


def test_models_registered_with_dummy_interface() -> None:
    """Test that model1, model2, model3, and model4 are registered with DummyModelInterface."""
    registry = PluginSystem.get_registered("DummyModelInterface")
    assert registry is not None

    plugin_names = registry.get_plugin_names()
    assert "DummyModel1" in plugin_names
    assert "DummyModel2" in plugin_names
    assert "CoupledModel" in plugin_names
    assert "MultiModel" in plugin_names
    assert len(plugin_names) == 4


def test_specs_accessible_by_name() -> None:
    """Each registered ModelSpec is findable by name via detect_specs()."""
    specs = DummyModelInterface.detect_specs()
    by_name = {s.name: s for s in specs}

    assert "DummyModel1" in by_name
    assert isinstance(by_name["DummyModel1"], ModelSpec)

    assert "DummyModel2" in by_name
    assert isinstance(by_name["DummyModel2"], ModelSpec)
