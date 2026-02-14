# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Free functions for initialization execution."""

import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .init_step import InitStep
from ..graph import (
    NetworkxTopologicalSorter,
    build_dependency_digraph,
    validate_dependency_graph,
)
from ..context import Context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InitGraphDiagnostic:
    """Machine-readable graph validation diagnostic."""

    code: Literal["duplicate_name", "missing_dependency", "cycle"]
    message: str
    step_name: Optional[str] = None
    dependency: Optional[str] = None
    cycle: tuple[str, ...] = ()


@dataclass(frozen=True)
class InitGraphValidationReport:
    """Validation report for an initialization dependency graph."""

    diagnostics: tuple[InitGraphDiagnostic, ...]

    @property
    def is_valid(self) -> bool:
        return not self.diagnostics


class InitializationGraphError(ValueError):
    """Raised when the initialization graph is invalid."""

    def __init__(self, report: InitGraphValidationReport):
        self.report = report
        message = report.diagnostics[0].message if report.diagnostics else ""
        super().__init__(message)


class InitGraphValidator:
    """Centralized validator for initialization dependency graphs."""

    @staticmethod
    def validate(lazy_inits: list[InitStep]) -> InitGraphValidationReport:
        node_names = [li.name for li in lazy_inits]
        dependencies_by_node = {li.name: li.depends_on for li in lazy_inits}
        report = validate_dependency_graph(node_names, dependencies_by_node)

        diagnostics = tuple(
            InitGraphDiagnostic(
                code=diag.code,
                message=diag.message,
                step_name=diag.node_name,
                dependency=diag.dependency,
                cycle=diag.cycle,
            )
            for diag in report.diagnostics
        )
        return InitGraphValidationReport(diagnostics=diagnostics)


def topological_sort(lazy_inits: list[InitStep]) -> list[InitStep]:
    """Sort lazy initializers by dependencies using DAG."""
    report = InitGraphValidator.validate(lazy_inits)
    if not report.is_valid:
        raise InitializationGraphError(report)

    name_to_init = {li.name: li for li in lazy_inits}
    graph = build_dependency_digraph({li.name: li.depends_on for li in lazy_inits})
    sorted_names = NetworkxTopologicalSorter().sort(graph)

    return [name_to_init[name] for name in sorted_names]


def execute_lazy_inits(lazy_inits: list[InitStep]) -> dict[str, Any]:
    """Execute lazy initializers in dependency order."""
    sorted_inits = topological_sort(lazy_inits)
    objects: dict[str, Any] = {}

    for lazy_init in sorted_inits:
        obj = lazy_init.execute(context=objects)
        objects[lazy_init.name] = obj

    return objects


def build_context_from_objects(objects: dict[str, Any]) -> Context:
    """Build Context from initialized objects."""
    fields = {}
    models = {}
    mesh = None
    runtime = None

    for name, obj in objects.items():
        if name.startswith("fields."):
            field_name = name.replace("fields.", "")
            fields[field_name] = obj
        elif name.startswith("operators."):
            operator_name = name.replace("operators.", "")
            models[operator_name] = obj
        elif name.startswith("models."):
            model_name = name.replace("models.", "")
            models[model_name] = obj
        elif name == "mesh":
            mesh = obj
        elif name == "runtime":
            runtime = obj
        else:
            logger.warning(
                "InitStep '%s' has no recognised prefix — routing to models", name
            )
            models[name] = obj

    return Context(fields=fields, models=models, mesh=mesh, runTime=runtime)


def execute_initialization(lazy_inits: list[InitStep]) -> Context:
    """Execute lazy initializers and build Context."""
    report = InitGraphValidator.validate(lazy_inits)
    if not report.is_valid:
        raise InitializationGraphError(report)

    objects = execute_lazy_inits(lazy_inits)
    return build_context_from_objects(objects)


def validate_lazy_init_graph(lazy_inits: list[InitStep]) -> list[tuple[str, str]]:
    """Validate lazy init graph for missing deps and cycles."""
    report = InitGraphValidator.validate(lazy_inits)
    errors: list[tuple[str, str]] = []

    for diag in report.diagnostics:
        if diag.code == "missing_dependency" and diag.step_name is not None:
            errors.append((diag.step_name, f"Missing dependency: {diag.dependency}"))
        elif diag.code in {"duplicate_name", "cycle"}:
            errors.append(("graph", diag.message))

    return errors
