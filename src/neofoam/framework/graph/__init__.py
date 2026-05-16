# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared graph utilities for DAG validation and ordering."""

from .dag_resolver import (
    DAGResolver,
    CyclicDependencyError,
    MissingDependencyError,
    collect_tagged_ops,
    infer_target_scope,
    build_global_graph,
    sort_global,
    rebuild_builder,
)
from .models import GraphDiagnostic, GraphValidationReport
from .operations_dag import (
    build_dag,
    build_global_dag,
    compute_nodes_order,
    compute_steps_order,
)
from .sorter import NetworkxTopologicalSorter, TopologicalSorter
from .validator import build_dependency_digraph, validate_dependency_graph
from .visualization import digraph_to_pyvis_html

__all__ = [
    "GraphDiagnostic",
    "GraphValidationReport",
    "TopologicalSorter",
    "NetworkxTopologicalSorter",
    "DAGResolver",
    "CyclicDependencyError",
    "MissingDependencyError",
    "collect_tagged_ops",
    "infer_target_scope",
    "build_global_graph",
    "sort_global",
    "rebuild_builder",
    "build_dependency_digraph",
    "build_dag",
    "build_global_dag",
    "compute_nodes_order",
    "compute_steps_order",
    "validate_dependency_graph",
    "digraph_to_pyvis_html",
]
