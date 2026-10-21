# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Validation utilities for dependency DAGs."""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional, Sequence, Tuple

import networkx as nx  # type: ignore[import-untyped]

from .sorter import NetworkxTopologicalSorter


@dataclass(frozen=True)
class _Diagnostic:
    """Machine-readable graph validation diagnostic (module-private)."""

    code: Literal["duplicate_name", "missing_dependency", "cycle"]
    message: str
    node_name: Optional[str] = None
    dependency: Optional[str] = None
    cycle: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphValidationReport:
    """Validation report for a dependency graph."""

    diagnostics: Tuple[_Diagnostic, ...]

    @property
    def is_valid(self) -> bool:
        return not self.diagnostics


def _build_dependency_digraph(
    dependencies_by_node: Mapping[str, Sequence[str]],
) -> nx.DiGraph:
    """Build a directed graph where edges are dependency -> dependant."""
    graph = nx.DiGraph()

    for node_name, dependencies in dependencies_by_node.items():
        graph.add_node(node_name)
        for dependency in dependencies:
            graph.add_edge(dependency, node_name)

    return graph


# Public alias kept for one release; will be hidden after callers migrate.
build_dependency_digraph = _build_dependency_digraph


Detector = Callable[
    [Sequence[str], Mapping[str, Sequence[str]]],
    list[_Diagnostic],
]


def _detect_duplicate_names(
    node_names: Sequence[str], _deps: Mapping[str, Sequence[str]]
) -> list[_Diagnostic]:
    diagnostics: list[_Diagnostic] = []
    name_counts = Counter(node_names)
    for node_name, count in sorted(name_counts.items()):
        if count > 1:
            diagnostics.append(
                _Diagnostic(
                    code="duplicate_name",
                    node_name=node_name,
                    message=f"Duplicate InitStep name detected: '{node_name}'",
                )
            )
    return diagnostics


def _detect_missing_dependencies(
    node_names: Sequence[str], dependencies_by_node: Mapping[str, Sequence[str]]
) -> list[_Diagnostic]:
    diagnostics: list[_Diagnostic] = []
    known_names = set(node_names)
    for node_name, dependencies in dependencies_by_node.items():
        for dependency in dependencies:
            if dependency in known_names:
                continue
            diagnostics.append(
                _Diagnostic(
                    code="missing_dependency",
                    node_name=node_name,
                    dependency=dependency,
                    message=(
                        f"InitStep '{node_name}' depends on '{dependency}', "
                        f"but '{dependency}' was not found"
                    ),
                )
            )
    return diagnostics


def _detect_cycles(
    _node_names: Sequence[str], dependencies_by_node: Mapping[str, Sequence[str]]
) -> list[_Diagnostic]:
    graph = _build_dependency_digraph(dependencies_by_node)
    try:
        NetworkxTopologicalSorter().sort(graph)
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        cycle_edges = nx.find_cycle(graph)
        cycle_names = tuple(edge[0] for edge in cycle_edges)
        return [
            _Diagnostic(
                code="cycle",
                cycle=cycle_names,
                message=f"Circular dependency: {' -> '.join(cycle_names)}",
            )
        ]
    return []


# Order matters: structural problems (duplicates, missing) gate cycle detection
# because a graph with missing edges may also appear acyclic for the wrong reason.
_DETECTORS: list[Detector] = [
    _detect_duplicate_names,
    _detect_missing_dependencies,
]


def validate_dependency_graph(
    node_names: Sequence[str], dependencies_by_node: Mapping[str, Sequence[str]]
) -> GraphValidationReport:
    """Validate a dependency graph and return structured diagnostics."""
    diagnostics: list[_Diagnostic] = []
    for detector in _DETECTORS:
        diagnostics.extend(detector(node_names, dependencies_by_node))

    # Cycle detection only runs when structural integrity is otherwise OK.
    if not diagnostics:
        diagnostics.extend(_detect_cycles(node_names, dependencies_by_node))

    return GraphValidationReport(diagnostics=tuple(diagnostics))
