# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Execute a :class:`StagedInitSpec`."""

from __future__ import annotations

from typing import Any, Optional

from ...context import Context
from ..config_context import ConfigContext
from ..execution import execute_initialization
from ..init_step import InitStep
from .spec import LoadResult, StagedInitSpec


class StagedInitRunner:
    """Run the LOAD → RESOLVE → BUILD pipeline of a frozen spec.

    Mutable state (``argv``, ``core_models``, ``optional_models``, ``state``)
    lives here, not on the spec — keeping the spec hashable and reusable.
    """

    def __init__(
        self, spec: StagedInitSpec, *, argv: Optional[list[str]] = None
    ) -> None:
        self._spec = spec
        self.argv: list[str] = argv or []
        self.core_models: list[Any] = []
        self.optional_models: list[Any] = []
        self.preprocess_tools: list[Any] = []
        self.state: Any = None

    @property
    def spec(self) -> StagedInitSpec:
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.name

    def run_load(self) -> LoadResult:
        if self._spec.load_fn is None:
            raise RuntimeError(f"No @{self._spec.name}.load defined")
        load_result = self._spec.load_fn()
        self.core_models = load_result.core_models
        self.optional_models = load_result.optional_models
        return load_result

    def run_resolve(self, config: ConfigContext) -> None:
        if self._spec.resolve_fn is not None:
            self._spec.resolve_fn(config)

    def run_build(self) -> list[InitStep]:
        if self._spec.build_fn is None:
            raise RuntimeError(f"No @{self._spec.name}.build defined")
        return self._spec.build_fn(self.core_models, self.optional_models)

    def run(self) -> Context:
        load_result = self.run_load()

        config = ConfigContext()
        for loaded_model in load_result.all_models:
            key = (
                getattr(loaded_model, "name", None)
                or type(loaded_model).__name__.lower()
            )
            if config.contains(key):
                raise ValueError(f"Duplicate model registration key: '{key}'")
            config.register(key, loaded_model)

        self.run_resolve(config)

        lazy_inits = self.run_build()
        return execute_initialization(lazy_inits)
