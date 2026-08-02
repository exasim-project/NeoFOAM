# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
ModelRuntime — one instantiation of a ModelSpec with its own loaded config.

Created by ModelSpec.instantiate(). Never shared between solver runs.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from neofoam.fields.synthesis import synthesize_init_step
from neofoam.io import BaseConfig

if TYPE_CHECKING:
    from neofoam.framework.initialization import ConfigContext, InitStep
    from neofoam.framework.operations import Operation

    from .spec import ModelSpec


@dataclass
class ModelRuntime:
    """
    One instantiation of a ModelSpec with its own loaded config.

    Owns per-instance state (config, name). All stage execution
    delegates to the spec, passing config explicitly so state never
    leaks between runs.
    """

    spec: "ModelSpec"
    name: str  # unique: "<spec.name>_<instance_id>"
    config: Any  # loaded config — updated during RESOLVE

    def run_resolve(self, ctx: "ConfigContext") -> None:
        """Call spec's resolve func; store the returned updated config."""
        if self.spec._resolve_func is not None:
            result = self.spec._resolve_func(self.config, ctx)
            if result is not None:
                self.config = result

    def run_build(self) -> list["InitStep"]:
        """Return the init-step list for this runtime.

        The framework auto-synthesizes one :class:`InitStep` per
        ``Model.field(...)`` declaration on ``self.spec`` — default
        factory ``<value_type>.read_field(mesh, name)``, with
        ``depends_on`` / ``write`` flowing from the declaration — and
        prepends them to whatever ``@spec.build`` returns. ``@build``
        is therefore reserved for everything the framework cannot
        synthesize: computed or intermediate fields, control objects,
        and side-effect-only steps that other declarations depend on.
        """
        if self.spec._build_func is None:
            user_steps: list["InitStep"] = []
        else:
            sig = inspect.signature(self.spec._build_func)
            user_steps = (
                self.spec._build_func(self.config)
                if len(sig.parameters) > 0
                else self.spec._build_func()
            )

        field_decls = getattr(self.spec, "_field_decls", None) or []
        if not field_decls:
            return user_steps

        auto_steps = [synthesize_init_step(decl) for decl in field_decls]
        return auto_steps + user_steps

    @property
    def operations(self) -> list["Operation"]:
        """Build operations with this runtime as the self binding."""
        return self.spec._build_operations_for(self)

    def native_operations(self) -> list["Operation"]:
        """Operations NOT tagged ``fallback=True`` (the model's native backend)."""
        return [op for op in self.operations if not op.metadata.fallback]

    def fallback_operations(self) -> list["Operation"]:
        """Operations tagged ``fallback=True`` (the model's fallback backend)."""
        return [op for op in self.operations if op.metadata.fallback]

    @property
    def configs(self) -> list[Any]:
        """
        Return BaseConfig instances held by this runtime.

        Used by ``LoadResult.configs`` to collect configs for validation.
        Handles both a single config and a SimpleNamespace of multiple configs.
        """
        result = []
        if isinstance(self.config, BaseConfig):
            result.append(self.config)
        elif isinstance(self.config, SimpleNamespace):
            for val in vars(self.config).values():
                if isinstance(val, BaseConfig):
                    result.append(val)
        return result
