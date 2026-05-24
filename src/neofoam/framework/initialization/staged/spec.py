# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Immutable spec + builder for the 3-stage init pipeline.

`StagedInitSpec` is the frozen description of one solver's load → resolve →
build pipeline (callbacks only, no state). `StagedInitSpecBuilder` collects
those callbacks via decorators and produces a `StagedInitSpec` on
`finalize()`. Execution lives in :mod:`runner`.

Note: this `*Spec` is per-solver-instance pipeline registration. It is
distinct from the long-lived ``ModelSpec`` / ``SolverSpec`` types that
declare what a solver *is*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..config_context import ConfigContext
from ..init_step import InitStep


@dataclass
class _ValidationError:
    """Module-private validation error (was the legacy public ``ValidationError``)."""

    field: str
    message: str
    severity: str = "error"


@dataclass
class LoadResult:
    """Outcome of the LOAD stage: core and optional model objects."""

    core_models: list[Any]
    optional_models: list[Any]

    @property
    def all_models(self) -> list[Any]:
        return self.core_models + self.optional_models

    @property
    def configs(self) -> list[Any]:
        configs: list[Any] = []
        for model in self.all_models:
            if hasattr(model, "configs"):
                configs.extend(model.configs)
        return configs

    @property
    def config_classes(self) -> list[type]:
        """Declared config schema set across all models.

        Returns the distinct ``BaseConfig`` subclasses every model
        declares — both classes registered via ``spec.config(...)`` (even
        when their instances are never loaded, e.g. a model whose
        ``@load`` short-circuits ``instantiate``) and the classes of any
        loaded config instances. Deduped by identity, order-preserved.

        This is the with-a-case view (after LOAD); for the case-free schema
        of a whole solver see ``collect_config_classes``.
        """
        from neofoam.io import collect_config_classes

        return collect_config_classes(self.all_models)

    def validate(self) -> list[Any]:
        """Validate all model configs and return a list of errors."""
        from neofoam.io import validate_models

        return validate_models(self.configs)


LoadFn = Callable[[], LoadResult]
ResolveFn = Callable[[ConfigContext], None]
BuildFn = Callable[[list[Any], list[Any]], list[InitStep]]


@dataclass(frozen=True)
class StagedInitSpec:
    """Immutable pipeline description ready to hand to a :class:`StagedInitRunner`."""

    name: str
    load_fn: Optional[LoadFn] = None
    resolve_fn: Optional[ResolveFn] = None
    build_fn: Optional[BuildFn] = None

    @classmethod
    def build(cls, name: str) -> StagedInitSpecBuilder:
        """Start a fluent registration of the three stages."""
        return StagedInitSpecBuilder(name)


class StagedInitSpecBuilder:
    """Decorator-driven registration of a :class:`StagedInitSpec`.

    Each decorator returns ``self`` so the resulting function reference is
    preserved on the user's side (the decorator pattern stays familiar).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._load_fn: Optional[LoadFn] = None
        self._resolve_fn: Optional[ResolveFn] = None
        self._build_fn: Optional[BuildFn] = None

    def load(self, fn: LoadFn) -> LoadFn:
        self._load_fn = fn
        return fn

    def resolve(self, fn: ResolveFn) -> ResolveFn:
        self._resolve_fn = fn
        return fn

    def build(self, fn: BuildFn) -> BuildFn:
        self._build_fn = fn
        return fn

    def finalize(self) -> StagedInitSpec:
        return StagedInitSpec(
            name=self._name,
            load_fn=self._load_fn,
            resolve_fn=self._resolve_fn,
            build_fn=self._build_fn,
        )
