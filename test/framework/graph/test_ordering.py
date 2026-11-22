# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the public ``topological_order`` dependency-ordering entry point.

Mirrors ``src/neofoam/framework/graph/ordering.py``: the empty mapping, a linear
dependency chain, tie-breaking through an injected sorter, and cycle propagation
from the underlying networkx backend.
"""

import networkx as nx
import pytest

from neofoam.framework.graph import NetworkxTopologicalSorter, topological_order


def test_topological_order_empty_map_returns_empty_list():
    assert topological_order({}) == []


def test_topological_order_linear_chain_orders_dependencies_first():
    order = topological_order({"B": ["A"], "C": ["B"], "A": []})

    assert order == ["A", "B", "C"]


def test_topological_order_uses_injected_sorter():
    priority = {"A": 2, "B": 0, "C": 1}
    sorter = NetworkxTopologicalSorter(key=lambda node: priority[node])

    order = topological_order({"A": [], "B": [], "C": []}, sorter=sorter)

    assert order == ["B", "C", "A"]


def test_topological_order_propagates_cycle_error():
    with pytest.raises(nx.NetworkXUnfeasible):
        topological_order({"A": ["B"], "B": ["A"]})
