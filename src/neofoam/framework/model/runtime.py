# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
ModelRuntime — one instantiation of a ModelSpec with its own loaded config.

Created by ModelSpec.instantiate(). Never shared between solver runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import ModelSpec
    from neofoam.framework.initialization import ConfigContext, InitStep
    from neofoam.framework.operations import Operations


@dataclass
class ModelRuntime:
    """
    One instantiation of a ModelSpec with its own loaded config.

    Owns per-instance state (config, name). All stage execution
    delegates to the spec, passing config explicitly so state never
    leaks between runs.
    """

    spec: "ModelSpec"
    name: str  # manifest "name" or spec.name for detect-only models
    config: Any  # loaded config — updated during RESOLVE

    def run_resolve(self, ctx: "ConfigContext") -> None:
        """Call spec's resolve func; store the returned updated config."""
        if self.spec._resolve_func is not None:
            result = self.spec._resolve_func(self.config, ctx)
            if result is not None:
                self.config = result

    def run_build(self) -> list["InitStep"]:
        """Call spec's build func with config and this runtime."""
        if self.spec._build_func is None:
            return []
        return self.spec._build_func(self.config, self)  # type: ignore[no-any-return]

    @property
    def operations(self) -> "Operations":
        """Build operations with this runtime as the self binding."""
        from neofoam.framework.operations import Operations

        return Operations(self.spec._build_operations_for(self))

    @property
    def configs(self) -> list[Any]:
        """
        Return BaseConfig instances held by this runtime.

        Used by ``LoadResult.configs`` to collect configs for validation.
        Handles both a single config and a SimpleNamespace of multiple configs.
        """
        from neofoam.io import BaseConfig
        from types import SimpleNamespace

        result = []
        if isinstance(self.config, BaseConfig):
            result.append(self.config)
        elif isinstance(self.config, SimpleNamespace):
            for val in vars(self.config).values():
                if isinstance(val, BaseConfig):
                    result.append(val)
        return result
