# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared graph utilities for DAG validation, ordering, and visualization.

Solvers and models in NeoFOAM declare their work as :class:`Operation`
objects with ``depends_on`` / ``before`` constraints. Before those
operations can run, the framework needs to:

1. Validate the dependency graph (no duplicates, no missing dependencies,
   no cycles) — see :mod:`~neofoam.framework.graph.validation`.
2. Resolve a topological order, respecting loop scopes and using
   ``operation_number`` as a tie-breaker — see
   :mod:`~neofoam.framework.graph.resolver`. For the simpler
   ``{node: [deps]}`` case, :func:`topological_order` (see
   :mod:`~neofoam.framework.graph.ordering`) builds the graph and sorts it.
3. Apply the graph edits that boundary conditions and models registered at
   runtime (``ctx.graph``) — see :mod:`~neofoam.framework.graph.edits`.
4. Optionally render the resulting graph for diagnostics — see
   :mod:`~neofoam.framework.graph.visualization`.

The package exposes its public symbols across six modules:

- :class:`DAGResolver` — the main entry point; merges and orders
  operations across loop scopes.
- :func:`topological_order` — order a plain ``{node: [deps]}`` mapping
  without touching networkx directly.
- :class:`TopologicalSorter` (Protocol) and
  :class:`NetworkxTopologicalSorter` (default impl) — the injection
  seam for swapping the sort backend.
- :func:`validate_dependency_graph`, :class:`GraphValidationReport`,
  and :func:`build_dependency_digraph` — pre-resolution graph
  validation and the underlying :class:`networkx.DiGraph` builder.
- :class:`CyclicDependencyError`, :class:`MissingDependencyError` —
  the two domain exceptions the resolver can raise.
- :class:`GraphEdits`, :class:`Selector` — runtime edits of a resolved
  graph, e.g. a loop around the operations that solve a field.
- :func:`dependency_dag`, :func:`digraph_to_pyvis_html` — compose
  per-domain DAGs and render them to an interactive HTML page.

See :doc:`/explanation/operations-and-the-dag` for the design
rationale and where the resolver fits into the solver lifecycle.
"""

from .edits import GraphEdits, Selector
from .ordering import topological_order
from .resolver import (
    CyclicDependencyError,
    DAGResolver,
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
    "GraphEdits",
    "Selector",
    "topological_order",
    "CyclicDependencyError",
    "MissingDependencyError",
    "build_dependency_digraph",
    "dependency_dag",
    "validate_dependency_graph",
    "digraph_to_pyvis_html",
]
