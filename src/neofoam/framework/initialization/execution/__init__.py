# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Execution layer for the lazy-init framework.

Given a list of
:class:`~neofoam.framework.initialization.init_step.InitStep`
objects, this layer:

1. Validates the dependency graph
   (:mod:`~neofoam.framework.initialization.execution.validation`).
2. Topologically sorts the steps using the graph package's sorter
   (:mod:`~neofoam.framework.initialization.execution.ordering`).
3. Executes each step in order, capturing results as
   :class:`InitResult`
   (:mod:`~neofoam.framework.initialization.execution.executor`).
4. Routes each result into the appropriate slot of a
   :class:`~neofoam.framework.context.Context` via a
   :class:`CategoryRouter`
   (:mod:`~neofoam.framework.initialization.execution.context_builder`).

The public entry point is :func:`execute_initialization` — one call
that runs all four stages and returns the populated Context. For
finer-grained control, the package re-exports the individual
building blocks (:func:`execute_step`, :func:`execute_lazy_inits`,
:func:`validate`, :class:`CategoryRouter`, :class:`InitResult`,
:class:`InitializationGraphError`).

The routing layer is open for extension: register a new category
handler with :meth:`CategoryRouter.register` and pass the router to
:func:`execute_initialization` to dispatch unknown categories
without modifying framework code.
"""

from __future__ import annotations

from ..init_step import InitStep
from ...context import Context
from .context_builder import (
    CategoryRouter,
    ContextBuilder,
    build_context_from_results,
    default_router,
)
from .executor import execute_lazy_inits, execute_step
from .init_result import InitResult
from .validation import InitializationGraphError, check_replacements, validate


def execute_initialization(
    lazy_inits: list[InitStep],
    *,
    router: CategoryRouter | None = None,
) -> Context:
    """Validate ``lazy_inits``, execute them in order, build a :class:`Context`.

    ``router`` defaults to :func:`default_router`; pass a custom router to
    extend category dispatch without touching the orchestrator.
    """
    report = validate(lazy_inits)
    if not report.is_valid:
        raise InitializationGraphError(report)
    # The replacement targets are enforced here (not only in ``_topological_sort``)
    # because the executed path sorts with ``validate_graph=False``; a real
    # ``replaces=[X]`` that names no present step must still raise.
    check_replacements(lazy_inits)

    results = execute_lazy_inits(lazy_inits, assume_validated=True)
    return build_context_from_results(results, router=router or default_router())


__all__ = [
    "CategoryRouter",
    "ContextBuilder",
    "InitResult",
    "InitializationGraphError",
    "build_context_from_results",
    "default_router",
    "execute_initialization",
    "execute_lazy_inits",
    "execute_step",
    "validate",
]
