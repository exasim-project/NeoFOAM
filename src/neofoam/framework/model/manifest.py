# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Manifest-based model loading from a YAML file.

Each manifest entry has ``type``, ``name``, and inline config fields.
The framework looks up the ModelSpec by type and calls instantiate().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from neofoam.core.plugin_system import PluginSystem

from .runtime import ModelRuntime


class ManifestError(Exception):
    """Raised when the manifest file is invalid."""


def load_manifest(
    manifest_path: Path,
    case_dir: Path,
    registry_name: str,
) -> list[ModelRuntime]:
    """Load models from a YAML manifest file.

    Args:
        manifest_path: Path to the YAML manifest file.
        case_dir: Base directory for model config resolution.
        registry_name: Name of the PluginSystem registry to look up specs.

    Returns:
        List of ModelRuntime instances, one per manifest entry.

    Raises:
        ManifestError: On validation failures (missing file, bad YAML, etc.).
    """
    if not manifest_path.exists():
        raise ManifestError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ManifestError(f"Manifest must be a YAML list, got {type(data).__name__}")

    # Validate type+name uniqueness
    seen: set[tuple[str, str]] = set()
    for entry in data:
        if not isinstance(entry, dict):
            raise ManifestError(
                f"Each manifest entry must be a dict, got {type(entry).__name__}"
            )
        if "type" not in entry or "name" not in entry:
            raise ManifestError(f"Manifest entry missing 'type' or 'name': {entry}")
        key = (entry["type"], entry["name"])
        if key in seen:
            raise ManifestError(
                f"Duplicate manifest entry: type={key[0]}, name={key[1]}"
            )
        seen.add(key)

    # Look up specs and instantiate
    registry = PluginSystem.get_registered(registry_name)
    if registry is None:
        raise ManifestError(f"No plugin registry found for '{registry_name}'")

    spec_by_type: dict[str, Any] = {}
    for plugin_cls in registry.plugin_registry:
        if hasattr(plugin_cls, "get_model_instance"):
            spec = plugin_cls.get_model_instance(plugin_cls)
            spec_by_type[spec.name] = spec

    runtimes: list[ModelRuntime] = []
    for entry in data:
        type_name = entry["type"]
        if type_name not in spec_by_type:
            raise ManifestError(f"Unknown model type '{type_name}' in manifest")
        spec = spec_by_type[type_name]
        rt = spec.instantiate(case_dir=case_dir, entry=entry)
        runtimes.append(rt)

    return runtimes
