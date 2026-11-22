# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Execute init steps in dependency order."""

from typing import Any

from neofoam import telemetry

from ..init_step import InitStep, InitStepExecutionError
from .init_result import InitResult
from .ordering import _topological_sort


def execute_step(step: InitStep, context: dict[str, Any]) -> Any:
    """Run one :class:`InitStep` against ``context`` and return its value.

    Wraps unexpected exceptions in :class:`InitStepExecutionError` to attach
    step name + declared dependencies. ``ValueError`` / ``TypeError`` raised
    by the initializer pass through unchanged.
    """
    if step.initializer is None:
        raise ValueError(f"InitStep '{step.name}' has no initializer function")

    try:
        with telemetry.span(f"init.{step.name}", category=step.category):
            return step.initializer(context)
    except (ValueError, TypeError):
        raise
    except Exception as exc:
        raise InitStepExecutionError(step.name, step.depends_on, exc) from exc


def execute_lazy_inits(
    lazy_inits: list[InitStep], *, assume_validated: bool = False
) -> list[InitResult]:
    """Execute ``lazy_inits`` in dependency order and collect their results."""
    sorted_inits = _topological_sort(lazy_inits, validate_graph=not assume_validated)
    context: dict[str, Any] = {}
    results: list[InitResult] = []

    with telemetry.span("initialization"):
        for lazy_init in sorted_inits:
            obj = execute_step(lazy_init, context)
            context[lazy_init.name] = obj
            results.append(
                InitResult(
                    name=lazy_init.name,
                    category=lazy_init.category,
                    value=obj,
                    write=lazy_init.write,
                )
            )

    return results
