# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Topological ordering of init steps; optional pre-validation."""

from __future__ import annotations

from ...graph import TopologicalSorter, topological_order
from ..init_step import InitStep
from .validation import InitializationGraphError, check_replacements, validate


def _topological_sort(
    lazy_inits: list[InitStep],
    *,
    validate_graph: bool = True,
    sorter: TopologicalSorter | None = None,
) -> list[InitStep]:
    """Sort ``lazy_inits`` by their declared dependencies.

    When ``validate_graph`` is true, the dependency graph is validated and
    :class:`InitializationGraphError` is raised on the first failure. The
    sorter defaults to :class:`NetworkxTopologicalSorter`.
    """
    if validate_graph:
        report = validate(lazy_inits)
        if not report.is_valid:
            raise InitializationGraphError(report)
        check_replacements(lazy_inits)

    name_to_init = {li.name: li for li in lazy_inits}
    sorted_names = topological_order(
        {li.name: li.depends_on for li in lazy_inits}, sorter=sorter
    )

    return [name_to_init[name] for name in sorted_names]
