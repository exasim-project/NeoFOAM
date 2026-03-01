# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Test that models are properly registered with DummyModelInterface.
"""

from pathlib import Path

from .models.dummy_model import DummyModelInterface
from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelSpec

CASE_DIR = Path(__file__).parent / "configs"


def test_detect_specs_returns_model_spec_objects() -> None:
    """detect_specs() must return (ModelSpec, DetectResult) tuples."""
    results = DummyModelInterface.detect_specs(case_dir=CASE_DIR)
    assert len(results) == 4
    assert all(isinstance(s, ModelSpec) for s, _ in results)
    names = {s.name for s, _ in results}
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
    results = DummyModelInterface.detect_specs(case_dir=CASE_DIR)
    by_name = {s.name: s for s, _ in results}

    assert "DummyModel1" in by_name
    assert isinstance(by_name["DummyModel1"], ModelSpec)

    assert "DummyModel2" in by_name
    assert isinstance(by_name["DummyModel2"], ModelSpec)


def test_multimodel_detect_returns_instance_ids() -> None:
    """MultiModel detect should return instance IDs from config file."""
    results = DummyModelInterface.detect_specs(case_dir=CASE_DIR)
    multi = next((s, dr) for s, dr in results if s.name == "MultiModel")
    spec, detect_result = multi
    assert detect_result.detected
    assert set(detect_result.instance_ids) == {"instance_a", "instance_b"}


def test_detect_specs_with_manifest_returns_runtimes() -> None:
    """detect_specs with manifest_path returns list[ModelRuntime] directly."""
    from neofoam.framework.model import ModelRuntime

    manifest_path = CASE_DIR / "models.yaml"
    runtimes = DummyModelInterface.detect_specs_with_manifest(
        case_dir=CASE_DIR, manifest_path=manifest_path
    )
    # Manifest covers MultiModel, detect covers the rest
    assert isinstance(runtimes, list)
    assert all(isinstance(rt, ModelRuntime) for rt in runtimes)
    names = {rt.spec.name for rt in runtimes}
    assert names == {"DummyModel1", "DummyModel2", "CoupledModel", "MultiModel"}
