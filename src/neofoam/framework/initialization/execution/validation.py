# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pre-execution validation of an :class:`InitStep` dependency graph."""

from ...graph import GraphValidationReport, validate_dependency_graph
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
