# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Unit tests for manifest-based model loading.
"""

from pathlib import Path

import pytest

from neofoam.framework.model import ModelSpec
from neofoam.framework.model.manifest import ManifestError, load_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REGISTRY_NAME = "TestManifestInterface"


def _register_spec(name: str, config_cls: type) -> ModelSpec:
    """Create and register a spec in the PluginSystem for testing."""
    from neofoam.core.plugin_system import PluginSystem

    spec = ModelSpec(name)
    spec.config(config_cls)

    # Register if not already done
    if PluginSystem.get_registered(REGISTRY_NAME) is None:
        from pydantic import BaseModel as BM

        @PluginSystem.register(
            discriminator_variable="model", discriminator="model_type"
        )
        class TestManifestInterface(BM):
            pass

    spec.register_with(PluginSystem._registry[REGISTRY_NAME].base_cls)
    return spec


# ===========================================================================
# Cycle 1 — load_manifest: valid YAML
# ===========================================================================


def test_load_manifest_returns_runtimes(tmp_path: Path) -> None:
    """load_manifest reads YAML and returns a list of ModelRuntime."""
    from neofoam.io import BaseConfig

    class TestCfg(BaseConfig):
        scale: float = 0.0

    _register_spec("ManifestTestModel", TestCfg)

    manifest = tmp_path / "models.yaml"
    manifest.write_text("- type: ManifestTestModel\n  name: inst_a\n  scale: 1.5\n")

    runtimes = load_manifest(manifest, tmp_path, REGISTRY_NAME)
    assert len(runtimes) == 1
    assert runtimes[0].name == "inst_a"
    assert runtimes[0].config.scale == 1.5


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    """load_manifest raises ManifestError if file doesn't exist."""
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.yaml", tmp_path, REGISTRY_NAME)


def test_load_manifest_invalid_yaml(tmp_path: Path) -> None:
    """load_manifest raises ManifestError if YAML is not a list."""
    manifest = tmp_path / "models.yaml"
    manifest.write_text("key: value\n")
    with pytest.raises(ManifestError, match="YAML list"):
        load_manifest(manifest, tmp_path, REGISTRY_NAME)


def test_load_manifest_duplicate_type_name(tmp_path: Path) -> None:
    """load_manifest raises ManifestError on duplicate type+name."""
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "- type: ManifestTestModel\n  name: dup\n  scale: 1.0\n"
        "- type: ManifestTestModel\n  name: dup\n  scale: 2.0\n"
    )
    with pytest.raises(ManifestError, match="Duplicate"):
        load_manifest(manifest, tmp_path, REGISTRY_NAME)


def test_load_manifest_unknown_type(tmp_path: Path) -> None:
    """load_manifest raises ManifestError for unknown model type."""
    manifest = tmp_path / "models.yaml"
    manifest.write_text("- type: NoSuchModel\n  name: inst\n")
    with pytest.raises(ManifestError, match="Unknown model type"):
        load_manifest(manifest, tmp_path, REGISTRY_NAME)


def test_load_manifest_missing_type_or_name(tmp_path: Path) -> None:
    """load_manifest raises ManifestError if entry lacks type or name."""
    manifest = tmp_path / "models.yaml"
    manifest.write_text("- name: only_name\n")
    with pytest.raises(ManifestError, match="'type' or 'name'"):
        load_manifest(manifest, tmp_path, REGISTRY_NAME)


def test_load_manifest_multiple_entries(tmp_path: Path) -> None:
    """load_manifest handles multiple entries of the same type."""
    from neofoam.io import BaseConfig

    class TestCfg2(BaseConfig):
        scale: float = 0.0

    _register_spec("ManifestMultiModel", TestCfg2)

    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "- type: ManifestMultiModel\n  name: a\n  scale: 1.0\n"
        "- type: ManifestMultiModel\n  name: b\n  scale: 2.0\n"
    )

    runtimes = load_manifest(manifest, tmp_path, REGISTRY_NAME)
    assert len(runtimes) == 2
    assert runtimes[0].name == "a"
    assert runtimes[1].name == "b"
    assert runtimes[0].config.scale == 1.0
    assert runtimes[1].config.scale == 2.0
