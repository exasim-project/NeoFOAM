# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What each region knows of a case's coupling, the checks of exchanges and nodes."""

from __future__ import annotations

import pytest
from framework.coupling.regions import contexts

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CouplingLoop,
    Exchange,
    InProcessCommChannel,
    NearestNeighborMapping,
    RegionCoupling,
    coupling,
)
from neofoam.framework.coupling.exchanges import region_couplings
from neofoam.framework.types import Placement

X = Exchange("x", (("a", "p"), ("b", "p")))


Y = Exchange("y", (("a", "q"), ("b", "q")))


def test_a_serial_node_enclosing_a_node_of_two_of_its_regions_raises() -> None:
    nodes = [CouplingLoop("fsi", ("x",), scheme="serial"), CouplingLoop("cht", ("y",))]

    with pytest.raises(
        ValueError,
        match=r"coupling 'fsi' is serial but encloses 'cht', which couples \['a', 'b'\] as "
        r"well; .* make 'fsi' parallel",
    ):
        region_couplings(["a", "b"], [X, Y], InProcessCommChannel(), nodes=nodes)


def test_a_serial_node_may_enclose_a_node_sharing_one_region() -> None:
    exchanges = [X, Exchange("z", (("b", "q"), ("c", "q")))]
    nodes = [CouplingLoop("ab", ("x",), scheme="serial"), CouplingLoop("bc", ("z",))]

    couplings = region_couplings(["a", "b", "c"], exchanges, InProcessCommChannel(), nodes=nodes)

    assert [node.scheme for node in couplings["b"].nodes] == ["serial", "serial"]


def test_a_node_is_serial_by_default() -> None:
    node = CouplingLoop("n", ("x", "y"))

    assert (node.scheme, node.parallel, node.placement_of("a")) == ("serial", False, Placement())


@pytest.mark.parametrize(
    ("nodes", "error"),
    [
        (
            [CouplingLoop("n", ("x",)), CouplingLoop("n", ("y",))],
            "two coupling nodes are named 'n'",
        ),
        (
            [CouplingLoop("n", ("x", "y")), CouplingLoop("m", ("y",))],
            "exchange 'y' is in the coupling nodes 'n' and 'm'",
        ),
        ([CouplingLoop("n", ("x",))], r"the exchanges \['y'\] are in no coupling node"),
        (
            [CouplingLoop("n", ("x", "w"))],
            "coupling node 'n' names the exchange 'w', which does not exist",
        ),
        (
            [CouplingLoop("n", ("x", "y"), placement={"c": Placement()})],
            r"coupling node 'n' places its op in the regions \['c'\], which it does not couple",
        ),
        (
            [
                CouplingLoop("n", ("x",), acceleration={"y": lambda s, i: s}),
                CouplingLoop("m", ("y",)),
            ],
            r"coupling node 'n' accelerates the exchanges \['y'\], which are not its own",
        ),
    ],
    ids=[
        "duplicate node name",
        "exchange in two nodes",
        "exchange in no node",
        "unknown exchange",
        "unknown region in placement",
        "acceleration of another exchange",
    ],
)
def test_region_couplings_check_the_nodes(nodes: list[CouplingLoop], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        region_couplings(["a", "b", "c"], [X, Y], InProcessCommChannel(), nodes=nodes)


def test_two_exchanges_of_one_name_raise() -> None:
    twin = Exchange("x", (("a", "q"), ("b", "q")))

    with pytest.raises(
        ValueError, match="two exchanges are named 'x'; an exchange's name is unique"
    ):
        region_couplings(
            ["a", "b"], [X, twin], InProcessCommChannel(), nodes=[CouplingLoop("n", ("x",))]
        )


def test_an_exchange_maps_by_nearest_neighbor_by_default() -> None:
    assert Exchange("x", (("a", "p"), ("b", "p"))).mapping == NearestNeighborMapping()


def _coupling(ctx: Context) -> RegionCoupling:
    region_coupling = coupling(ctx).coupling
    assert region_coupling is not None
    return region_coupling


def test_unconnected_groups_of_exchanges_get_a_leader_each() -> None:
    exchanges = [
        Exchange("a-b", (("a", "p"), ("b", "p"))),
        Exchange("c-d", (("c", "p"), ("d", "p"))),
    ]
    regions = contexts(list("abcd"), exchanges)

    couplings = [_coupling(ctx) for ctx in regions.values()]
    assert [(coupling.group, coupling.leader) for coupling in couplings] == [
        (("a", "b"), "b"),
        (("a", "b"), "b"),
        (("c", "d"), "d"),
        (("c", "d"), "d"),
    ]


@pytest.mark.parametrize(
    ("exchange", "error"),
    [
        (Exchange("x", (("a", "p"), ("fluid", "q"))), r"couple the regions \['fluid'\]"),
        (Exchange("x", (("a", "p"), ("a", "q"))), "exchange 'x' joins two ports of region a"),
    ],
)
def test_region_couplings_check_the_links(exchange: Exchange, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        region_couplings(
            ["a", "b"], [exchange], InProcessCommChannel(), [CouplingLoop("n", ("x",))]
        )


def test_a_patch_is_in_one_exchange() -> None:
    exchanges = [Exchange("x", (("a", "p"), ("b", "p"))), Exchange("y", (("a", "p"), ("b", "q")))]

    with pytest.raises(ValueError, match="a.p is in the exchanges 'x' and 'y'"):
        region_couplings(
            ["a", "b"], exchanges, InProcessCommChannel(), [CouplingLoop("n", ("x", "y"))]
        )


def test_a_region_no_exchange_names_is_not_coupled() -> None:
    couplings = region_couplings(
        ["a", "b", "c"], [X], InProcessCommChannel(), [CouplingLoop("n", ("x",))]
    )

    assert list(couplings) == ["a", "b"]
