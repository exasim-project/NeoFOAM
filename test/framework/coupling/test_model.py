# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A region's coupling model: which sides it takes and the coupling ops it builds of them."""

from __future__ import annotations

import pytest
from framework.coupling.regions import add_coupled, contexts, fixed, graph, solve

from neofoam.framework.context import Context
from neofoam.framework.coupling import CouplingLoop, Exchange, IdentityMapping, Patch, coupling
from neofoam.framework.coupling.iteration._loop_op import CouplingLoopOp
from neofoam.framework.types import Placement

EXCHANGE = Exchange("a-b", (("a", "p"), ("b", "p")), IdentityMapping())
X = Exchange("x", (("a", "p"), ("b", "p")))
Y = Exchange("y", (("a", "q"), ("b", "q")))


def test_the_coupling_model_builds_one_coupling_op_once() -> None:
    exchanges = [EXCHANGE, Exchange("a-b-2", (("a", "q"), ("b", "q")), IdentityMapping())]
    regions = contexts(["a", "b"], exchanges)
    add_coupled(regions["a"], "p", fixed(data="x"))
    add_coupled(regions["a"], "q", fixed(data="y"))

    model = coupling(regions["a"])
    (couple_op,) = model.operations()

    assert model.operations() is model.operations()
    assert isinstance(couple_op.func, CouplingLoopOp)
    assert couple_op.metadata.wraps == Placement()
    assert [port.name for port in couple_op.func.ports] == ["p", "q"]
    with pytest.raises(ValueError, match="a.p is added after the region's graph was built"):
        add_coupled(regions["a"], "p", fixed(data="x"))


def test_a_region_whose_solver_added_no_coupling_model_cannot_couple() -> None:
    with pytest.raises(ValueError, match="the solver of region a does not support coupling"):
        coupling(Context(fields={}, models={}, region="a"))


def test_a_region_the_case_does_not_couple_has_no_coupling_ops() -> None:
    regions = contexts(["a", "b", "c"], [EXCHANGE])

    assert len(coupling(regions["c"]).operations()) == 0
    (loop,) = graph(regions["c"], solve("x", lambda ctx: None)).ops
    assert loop.operation_name == "solve_x"


def test_a_port_in_a_region_the_case_does_not_couple_raises() -> None:
    regions = contexts(["a", "b", "c"], [EXCHANGE])

    with pytest.raises(ValueError, match="the case does not couple its region"):
        add_coupled(regions["c"], "p", fixed(data="x"))


def test_a_port_no_exchange_names_raises() -> None:
    regions = contexts(["a", "b"], [EXCHANGE])

    with pytest.raises(ValueError, match=r"no exchange connects the port a\.top"):
        add_coupled(regions["a"], "top", fixed(data="x"))


def test_only_a_coupled_instance_is_added() -> None:
    model = coupling(contexts(["a", "b"], [EXCHANGE])["a"])

    with pytest.raises(TypeError, match="object is not @coupled"):
        model.add(Patch("p"), object())


def test_a_region_s_data_field_is_in_at_most_one_node() -> None:
    nodes = [CouplingLoop("n", ("x",), scheme="parallel"), CouplingLoop("m", ("y",))]
    regions = contexts(["a", "b"], [X, Y], nodes)
    add_coupled(regions["a"], "p", fixed(data="T"))
    add_coupled(regions["a"], "q", fixed(data="T"))

    with pytest.raises(
        ValueError, match="the data T of region a is coupled in the nodes 'n' and 'm'"
    ):
        coupling(regions["a"]).operations()
