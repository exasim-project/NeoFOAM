# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A region's coupling model: which sides it takes and the coupling ops it builds of them."""

from __future__ import annotations

import pytest
from framework.coupling.regions import add_coupled, contexts, fixed, graph, solve

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling, CouplingNode, Exchange, IdentityMapping
from neofoam.framework.coupling.loop import CouplingOp
from neofoam.framework.types import Placement

EXCHANGE = Exchange("a-b", (("a", "p"), ("b", "p")), IdentityMapping())
X = Exchange("x", (("a", "p"), ("b", "p")))
Y = Exchange("y", (("a", "q"), ("b", "q")))


def test_the_coupling_model_builds_one_coupling_op_once() -> None:
    exchanges = [EXCHANGE, Exchange("a-b-2", (("a", "q"), ("b", "q")), IdentityMapping())]
    regions = contexts(["a", "b"], exchanges)
    add_coupled(regions["a"], "p", "x", fixed())
    add_coupled(regions["a"], "q", "y", fixed())

    model = Coupling.of(regions["a"])
    (couple_op,) = model.operations()

    assert model.operations() is model.operations()
    assert isinstance(couple_op.func, CouplingOp)
    assert couple_op.metadata.wraps == Placement()
    assert [side.patch for side in couple_op.func.sides] == ["p", "q"]
    with pytest.raises(ValueError, match="a.p is added after the region's graph was built"):
        add_coupled(regions["a"], "p", "x", fixed())


def test_a_region_whose_solver_added_no_coupling_model_cannot_couple() -> None:
    with pytest.raises(ValueError, match="the solver of region a does not support coupling"):
        Coupling.of(Context(fields={}, models={}, region="a"))


def test_a_region_the_case_does_not_couple_has_no_coupling_ops() -> None:
    regions = contexts(["a", "b", "c"], [EXCHANGE])

    assert len(Coupling.of(regions["c"]).operations()) == 0
    (loop,) = graph(regions["c"], solve("x", lambda ctx: None)).ops
    assert loop.operation_name == "solve_x"


def test_a_side_in_a_region_the_case_does_not_couple_raises() -> None:
    regions = contexts(["a", "b", "c"], [EXCHANGE])

    with pytest.raises(ValueError, match="the case does not couple its region"):
        add_coupled(regions["c"], "p", "x", fixed())


def test_a_side_on_a_patch_no_exchange_names_raises() -> None:
    regions = contexts(["a", "b"], [EXCHANGE])

    with pytest.raises(ValueError, match=r"no exchange connects the coupled side of a\.top"):
        add_coupled(regions["a"], "top", "x", fixed())


def test_only_a_coupled_instance_with_field_and_patch_is_added() -> None:
    model = Coupling.of(contexts(["a", "b"], [EXCHANGE])["a"])

    with pytest.raises(TypeError, match="object is not @coupled"):
        model.add(object())


def test_a_region_s_field_is_in_at_most_one_node() -> None:
    nodes = [CouplingNode("n", ("x",), scheme="parallel"), CouplingNode("m", ("y",))]
    regions = contexts(["a", "b"], [X, Y], nodes)
    add_coupled(regions["a"], "p", "T", fixed())
    add_coupled(regions["a"], "q", "T", fixed())

    with pytest.raises(ValueError, match="field T of region a is coupled in the nodes 'n' and 'm'"):
        Coupling.of(regions["a"]).operations()
