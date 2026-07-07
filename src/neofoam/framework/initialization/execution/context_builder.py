# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Route :class:`InitResult` values into a :class:`Context`.

The routing layer is open for extension: register additional category
handlers via :class:`CategoryRouter.register`. Unknown categories fall
through to ``models`` with a logged warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ...context import Context
from .init_result import InitResult


logger = logging.getLogger(__name__)


def _strip_prefix(name: str, prefix: str) -> str:
    if name.startswith(f"{prefix}."):
        return name[len(prefix) + 1 :]
    return name


@dataclass
class ContextBuilder:
    """Mutable accumulator routed values are written into.

    Calling :meth:`to_context` produces the immutable :class:`Context`.
    """

    fields: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    mesh: Any = None
    time: Any = None
    write_fields: set[str] = field(default_factory=set)

    def to_context(self) -> Context:
        return Context(
            fields=self.fields,
            models=self.models,
            mesh=self.mesh,
            time=self.time,
            write_fields=self.write_fields,
        )


Handler = Callable[[ContextBuilder, str, Any], None]


def _route_fields(builder: ContextBuilder, name: str, value: Any) -> None:
    builder.fields[_strip_prefix(name, "fields")] = value


def _route_models(builder: ContextBuilder, name: str, value: Any) -> None:
    builder.models[_strip_prefix(name, "models")] = value


def _route_operators(builder: ContextBuilder, name: str, value: Any) -> None:
    builder.models[_strip_prefix(name, "operators")] = value


def _route_resource(builder: ContextBuilder, name: str, value: Any) -> None:
    if name == "mesh":
        builder.mesh = value
    elif name == "time":
        builder.time = value
    else:
        _fallback_to_models(builder, name, value)


def _fallback_to_models(builder: ContextBuilder, name: str, value: Any) -> None:
    logger.warning(
        "InitStep '%s' has no special context slot — routing to models",
        name,
    )
    builder.models[name] = value


class CategoryRouter:
    """Registry mapping :attr:`InitResult.category` to handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, category: str, handler: Handler) -> None:
        self._handlers[category] = handler

    def route(self, builder: ContextBuilder, result: InitResult) -> None:
        # init-only resources (leading underscore) stay in the executor's
        # working dict for dependent build steps but are never routed onto the
        # Context — e.g. a backend Foam::Time held only by the mesh/hook/backend.
        if result.name.startswith("_"):
            return
        handler = self._handlers.get(result.category, _fallback_to_models)
        handler(builder, result.name, result.value)
        # collect fields flagged for persistence (auto-write)
        if result.write:
            builder.write_fields.add(_strip_prefix(result.name, "fields"))


def default_router() -> CategoryRouter:
    """Router populated with the four built-in category handlers."""
    router = CategoryRouter()
    router.register("fields", _route_fields)
    router.register("models", _route_models)
    router.register("operators", _route_operators)
    router.register("resource", _route_resource)
    return router


def build_context_from_results(
    init_results: list[InitResult],
    *,
    router: CategoryRouter | None = None,
) -> Context:
    """Build a :class:`Context` by routing ``init_results`` through ``router``."""
    builder = ContextBuilder()
    active_router = router or default_router()
    for result in init_results:
        active_router.route(builder, result)
    return builder.to_context()
