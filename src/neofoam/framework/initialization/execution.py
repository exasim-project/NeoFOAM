# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Free functions for initialization execution."""

import logging
from dataclasses import dataclass
from typing import Any

from .init_step import InitCategory, InitStep
from ..graph import (
    GraphValidationReport,
    NetworkxTopologicalSorter,
    build_dependency_digraph,
    validate_dependency_graph,
)
from ..context import Context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InitResult:
    """Result of executing a single InitStep.

    Carries the step's ``name``, ``category``, and the produced ``value``
    so that downstream routing can use the category instead of parsing
    the name prefix.
    """

    name: str
    category: InitCategory
    value: Any


class InitializationGraphError(ValueError):
    """Raised when the initialization graph is invalid."""

    def __init__(self, report: GraphValidationReport):
        self.report = report
        message = report.diagnostics[0].message if report.diagnostics else ""
        super().__init__(message)


def validate(lazy_inits: list[InitStep]) -> GraphValidationReport:
    """Validate initialization dependency graph."""
    node_names = [li.name for li in lazy_inits]
    dependencies_by_node = {li.name: li.depends_on for li in lazy_inits}
    return validate_dependency_graph(node_names, dependencies_by_node)


def topological_sort(
    lazy_inits: list[InitStep], *, validate_graph: bool = True
) -> list[InitStep]:
    """Sort lazy initializers by dependencies using DAG.

    Args:
        lazy_inits: Init steps to sort.
        validate_graph: Whether to validate the dependency graph before sorting.
    """
    if validate_graph:
        report = validate(lazy_inits)
        if not report.is_valid:
            raise InitializationGraphError(report)

    name_to_init = {li.name: li for li in lazy_inits}
    graph = build_dependency_digraph({li.name: li.depends_on for li in lazy_inits})
    sorted_names = NetworkxTopologicalSorter().sort(graph)

    return [name_to_init[name] for name in sorted_names]


def execute_lazy_inits(
    lazy_inits: list[InitStep], *, assume_validated: bool = False
) -> list[InitResult]:
    """Execute lazy initializers in dependency order.

    Returns a list of :class:`InitResult` objects preserving both the
    step ``category`` and the produced ``value``.
    """
    sorted_inits = topological_sort(lazy_inits, validate_graph=not assume_validated)
    context: dict[str, Any] = {}
    results: list[InitResult] = []

    for lazy_init in sorted_inits:
        obj = lazy_init.execute(context=context)
        context[lazy_init.name] = obj
        results.append(
            InitResult(
                name=lazy_init.name,
                category=lazy_init.category,
                value=obj,
            )
        )

    return results


def _strip_prefix(name: str, prefix: str) -> str:
    """Remove *prefix.* from *name* if present."""
    if name.startswith(f"{prefix}."):
        return name[len(prefix) + 1 :]
    return name


def build_context_from_objects(init_results: list[InitResult]) -> Context:
    """Build Context from executed InitResults.

    Routing strategy is category-driven.
    """
    fields: dict[str, Any] = {}
    models: dict[str, Any] = {}
    mesh: Any = None
    runtime: Any = None

    for result in init_results:
        name = result.name
        category = result.category
        obj = result.value

        if category == "fields":
            fields[_strip_prefix(name, "fields")] = obj
        elif category in {"operators", "models"}:
            models[_strip_prefix(name, category)] = obj
        elif category == "resource" and name == "mesh":
            mesh = obj
        elif category == "resource" and name == "runtime":
            runtime = obj
        else:
            logger.warning(
                "InitStep '%s' with category '%s' has no special context slot "
                "— routing to models",
                name,
                category,
            )
            models[name] = obj

    return Context(fields=fields, models=models, mesh=mesh, runtime=runtime)


def execute_initialization(lazy_inits: list[InitStep]) -> Context:
    """Execute lazy initializers and build Context."""
    report = validate(lazy_inits)
    if not report.is_valid:
        raise InitializationGraphError(report)

    init_results = execute_lazy_inits(lazy_inits, assume_validated=True)
    return build_context_from_objects(init_results)
