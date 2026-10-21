# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pre-execution validation of an :class:`InitStep` dependency graph."""

from ...graph import GraphValidationReport, validate_dependency_graph
from ...graph.validation import _Diagnostic
from ..init_step import InitStep


class InitializationGraphError(ValueError):
    """Raised when an initialization dependency graph fails validation."""

    def __init__(self, report: GraphValidationReport):
        self.report = report
        message = report.diagnostics[0].message if report.diagnostics else ""
        super().__init__(message)


def validate(lazy_inits: list[InitStep]) -> GraphValidationReport:
    """Run the graph validators against ``lazy_inits``."""
    node_names = [li.name for li in lazy_inits]
    dependencies_by_node = {li.name: li.depends_on for li in lazy_inits}
    return validate_dependency_graph(node_names, dependencies_by_node)


def check_replacements(lazy_inits: list[InitStep]) -> None:
    """Raise if any ``replaces=[X]`` names a step that is not present.

    A step may carry its own name in ``replaces=`` (the terminal-alias
    pattern), which self-resolves; only a target with no matching step is an
    error.
    """
    present = {li.name for li in lazy_inits}
    for li in lazy_inits:
        for target in li.replaces:
            if target in present:
                continue
            report = GraphValidationReport(
                diagnostics=(
                    _Diagnostic(
                        code="missing_dependency",
                        node_name=li.name,
                        dependency=target,
                        message=(
                            f"InitStep '{li.name}' declares replaces="
                            f"['{target}'] but no step named '{target}' exists"
                        ),
                    ),
                )
            )
            raise InitializationGraphError(report)
