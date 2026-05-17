# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared graph utilities for DAG validation and ordering."""

from .resolver import (
    DAGResolver,
    CyclicDependencyError,
    MissingDependencyError,
)
from .sorter import NetworkxTopologicalSorter, TopologicalSorter
from .validation import (
    GraphValidationReport,
    build_dependency_digraph,
    validate_dependency_graph,
)
from .visualization import dependency_dag, digraph_to_pyvis_html

__all__ = [
    "GraphValidationReport",
    "TopologicalSorter",
    "NetworkxTopologicalSorter",
    "DAGResolver",
    "CyclicDependencyError",
    "MissingDependencyError",
    "build_dependency_digraph",
    "dependency_dag",
    "validate_dependency_graph",
    "digraph_to_pyvis_html",
]
