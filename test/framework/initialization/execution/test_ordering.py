# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for topological ordering of init steps by dependency."""

import pytest

from neofoam.framework.initialization.execution.ordering import _topological_sort
from neofoam.framework.initialization.init_step import InitStep


def test_sort_linear_chain(linear_chain):
    names = [li.name for li in _topological_sort(linear_chain)]
    assert names == ["A", "B", "C"]


def test_sort_independent():
    inits = [
        InitStep("X", depends_on=[], initializer=lambda _ctx: "x"),
        InitStep("Y", depends_on=[], initializer=lambda _ctx: "y"),
        InitStep("Z", depends_on=[], initializer=lambda _ctx: "z"),
    ]
    names = {li.name for li in _topological_sort(inits)}
    assert names == {"X", "Y", "Z"}


def test_sort_diamond(diamond_graph):
    names = [li.name for li in _topological_sort(diamond_graph)]
    assert names[0] == "A"
    assert names[-1] == "D"
    assert set(names[1:3]) == {"B", "C"}


def test_cycle_raises():
    inits = [
        InitStep("A", depends_on=["C"], initializer=lambda _ctx: "a"),
        InitStep("B", depends_on=["A"], initializer=lambda _ctx: "b"),
        InitStep("C", depends_on=["B"], initializer=lambda _ctx: "c"),
    ]
    with pytest.raises(ValueError, match="Circular dependency"):
        _topological_sort(inits)


def test_missing_dep_raises():
    inits = [InitStep("A", depends_on=["missing"], initializer=lambda _ctx: "a")]
    with pytest.raises(ValueError, match="depends on 'missing'"):
        _topological_sort(inits)


def test_self_dependency_raises():
    inits = [InitStep("A", depends_on=["A"], initializer=lambda _ctx: "a")]
    with pytest.raises(ValueError, match="Circular dependency"):
        _topological_sort(inits)


def test_duplicate_name_raises():
    inits = [
        InitStep("A", depends_on=[], initializer=lambda _ctx: "a1"),
        InitStep("A", depends_on=[], initializer=lambda _ctx: "a2"),
    ]
    with pytest.raises(ValueError, match="Duplicate InitStep name"):
        _topological_sort(inits)
