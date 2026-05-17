# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Execution layer for the lazy-init framework.

Public entry point is :func:`execute_initialization`. The package also
re-exports :class:`InitResult`, :class:`InitializationGraphError`,
:class:`CategoryRouter`, and :func:`execute_step` for callers that need
finer-grained access.
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
from .validation import InitializationGraphError, validate


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
