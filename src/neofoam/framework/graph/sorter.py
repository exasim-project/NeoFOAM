# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Topological sort abstractions and implementations."""

from typing import Any, Callable, Optional, Protocol

import networkx as nx  # type: ignore[import-untyped]


class TopologicalSorter(Protocol):
    """Interface boundary for DAG sorting backends."""

    def sort(
        self,
        graph: nx.DiGraph,
        key: Optional[Callable[[str], Any]] = None,
    ) -> list[str]:
        """Return nodes in valid topological order, with optional tie-break key."""


class NetworkxTopologicalSorter:
    """Topological sorter implementation based on networkx."""

    def __init__(self, key: Optional[Callable[[str], Any]] = None):
        self._key = key

    def sort(
        self,
        graph: nx.DiGraph,
        key: Optional[Callable[[str], Any]] = None,
    ) -> list[str]:
        """Return nodes in topological order.

        Per-call ``key`` overrides the instance-level key; otherwise the
        instance key is used; otherwise sort lexicographically by node name.
        """
        sort_key = key or self._key or (lambda node_name: node_name)
        return list(nx.lexicographical_topological_sort(graph, key=sort_key))
