# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Public dependency-ordering entry point.

Consumers that have a ``{node: [dependencies]}`` mapping and only need a valid
execution order should call :func:`topological_order` rather than building a
:class:`networkx.DiGraph` and invoking a sorter themselves. Keeping the graph
construction here ensures all DAG handling lives in
:mod:`neofoam.framework.graph`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .sorter import NetworkxTopologicalSorter, TopologicalSorter
from .validation import _build_dependency_digraph


def topological_order(
    dependencies_by_node: Mapping[str, Sequence[str]],
    *,
    sorter: TopologicalSorter | None = None,
) -> list[str]:
    """Return node names in dependency order (dependencies before dependants).

    Builds the dependency digraph and delegates the ordering to ``sorter``
    (default :class:`NetworkxTopologicalSorter`). The caller is responsible for
    validating the graph first if it needs structured diagnostics — see
    :func:`validate_dependency_graph`.
    """
    graph = _build_dependency_digraph(dependencies_by_node)
    active_sorter = sorter or NetworkxTopologicalSorter()
    return active_sorter.sort(graph)
