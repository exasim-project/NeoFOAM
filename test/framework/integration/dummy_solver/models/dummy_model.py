# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Base class for DummySolver models.

Mimics SimpleSolverModel structure for testing.
"""

from typing import Optional
from pathlib import Path

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelSpec, ModelRuntime, DetectResult, load_manifest


def Model(name: str) -> ModelSpec:
    """Factory for dummy models using the ModelSpec API."""
    return ModelSpec(name)


@PluginSystem.register(discriminator_variable="model", discriminator="model_type")
class DummyModelInterface(BaseModel):
    """
    Base class for DummySolver optional models.

    Provides infrastructure for automatic model detection and integration.
    """

    @classmethod
    def detect_specs(
        cls, case_dir: Optional[Path] = None
    ) -> list[tuple[ModelSpec, DetectResult]]:
        """
        Return (spec, DetectResult) pairs for all detected models.

        DetectResult.instance_ids may contain multiple IDs for multi-instance models.
        """
        registry = PluginSystem.get_registered("DummyModelInterface")
        if not registry:
            return []

        results: list[tuple[ModelSpec, DetectResult]] = []
        for plugin_cls in registry.plugin_registry:
            if not hasattr(plugin_cls, "get_model_instance"):
                continue
            spec = plugin_cls.get_model_instance(plugin_cls)
            detect_result = spec.run_detect(case_dir=case_dir)
            if detect_result.detected:
                results.append((spec, detect_result))
        return results

    @classmethod
    def detect_specs_with_manifest(
        cls,
        case_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ) -> list[ModelRuntime]:
        """
        Load models from manifest, then auto-detect any specs not already covered.

        1. Load manifest entries → list[ModelRuntime]
        2. Auto-detect specs not already covered by manifest type names
        3. Return combined list
        """
        runtimes: list[ModelRuntime] = []
        manifest_types: set[str] = set()

        # 1. Load from manifest
        if manifest_path is not None and manifest_path.exists():
            manifest_rts = load_manifest(
                manifest_path, case_dir or Path("."), "DummyModelInterface"
            )
            runtimes.extend(manifest_rts)
            manifest_types = {rt.spec.name for rt in manifest_rts}

        # 2. Auto-detect specs not covered by manifest
        for spec, detect_result in cls.detect_specs(case_dir=case_dir):
            if spec.name in manifest_types:
                continue
            if detect_result.instance_ids:
                for iid in detect_result.instance_ids:
                    entry = {"type": spec.name, "name": iid}
                    runtimes.append(
                        spec.instantiate(case_dir=case_dir or Path("."), entry=entry)
                    )
            else:
                runtimes.append(spec.instantiate(case_dir=case_dir or Path(".")))

        return runtimes
