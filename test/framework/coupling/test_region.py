# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What each region knows of a case's coupling, the checks of links and nodes, and run_regions."""

from __future__ import annotations

import pytest
from framework.coupling.regions import TIMEOUT, contexts

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    Coupling,
    CouplingDeadlock,
    CouplingNode,
    InProcessTransport,
    Link,
    NearestNeighborMapping,
    RegionCoupling,
    TransportClosed,
)
from neofoam.framework.coupling.region import region_couplings, run_regions
from neofoam.framework.types import Placement

X = Link("x", (("a", "p"), ("b", "p")))
Y = Link("y", (("a", "q"), ("b", "q")))


def test_a_serial_node_enclosing_a_node_of_two_of_its_regions_raises() -> None:
    nodes = [CouplingNode("fsi", ("x",), scheme="serial"), CouplingNode("cht", ("y",))]

    with pytest.raises(
        ValueError,
        match=r"coupling 'fsi' is serial but encloses 'cht', which couples \['a', 'b'\] as "
        r"well; .* make 'fsi' parallel",
    ):
        region_couplings(["a", "b"], [X, Y], InProcessTransport(), nodes=nodes)


def test_a_serial_node_may_enclose_a_node_sharing_one_region() -> None:
    links = [X, Link("z", (("b", "q"), ("c", "q")))]
    nodes = [CouplingNode("ab", ("x",), scheme="serial"), CouplingNode("bc", ("z",))]

    couplings = region_couplings(["a", "b", "c"], links, InProcessTransport(), nodes=nodes)

    assert [node.scheme for node in couplings["b"].nodes] == ["serial", "serial"]


def test_a_node_is_serial_by_default() -> None:
    couplings = region_couplings(["a", "b"], [X, Y], InProcessTransport())

    (node,) = couplings["a"].nodes
    assert (node.name, node.scheme, node.parallel) == ("coupling", "serial", False)


def test_the_default_node_holds_every_exchange() -> None:
    couplings = region_couplings(["a", "b"], [X, Y], InProcessTransport())

    (node,) = couplings["a"].nodes
    assert (node.name, node.exchanges, node.placement_of("a")) == (
        "coupling",
        ("x", "y"),
        Placement(),
    )


@pytest.mark.parametrize(
    ("nodes", "error"),
    [
        (
            [CouplingNode("n", ("x",)), CouplingNode("n", ("y",))],
            "two coupling nodes are named 'n'",
        ),
        (
            [CouplingNode("n", ("x", "y")), CouplingNode("m", ("y",))],
            "exchange 'y' is in the coupling nodes 'n' and 'm'",
        ),
        ([CouplingNode("n", ("x",))], r"the exchanges \['y'\] are in no coupling node"),
        (
            [CouplingNode("n", ("x", "w"))],
            "coupling node 'n' names the exchange 'w', which does not exist",
        ),
        (
            [CouplingNode("n", ("x", "y"), placement={"c": Placement()})],
            r"coupling node 'n' places its op in the regions \['c'\], which it does not couple",
        ),
        (
            [
                CouplingNode("n", ("x",), acceleration={"y": lambda s, i: s}),
                CouplingNode("m", ("y",)),
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
def test_region_couplings_check_the_nodes(nodes: list[CouplingNode], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        region_couplings(["a", "b", "c"], [X, Y], InProcessTransport(), nodes=nodes)


def test_two_exchanges_of_one_name_raise() -> None:
    twin = Link("x", (("a", "q"), ("b", "q")))

    with pytest.raises(
        ValueError, match="two exchanges are named 'x'; an exchange's name is unique"
    ):
        region_couplings(
            ["a", "b"], [X, twin], InProcessTransport(), nodes=[CouplingNode("n", ("x",))]
        )


def test_an_exchange_maps_by_nearest_neighbor_by_default() -> None:
    assert Link("x", (("a", "p"), ("b", "p"))).mapping == NearestNeighborMapping()


def _coupling(ctx: Context) -> RegionCoupling:
    coupling = Coupling.of(ctx).coupling
    assert coupling is not None
    return coupling


def test_unconnected_groups_of_exchanges_get_a_leader_each() -> None:
    links = [Link("a-b", (("a", "p"), ("b", "p"))), Link("c-d", (("c", "p"), ("d", "p")))]
    regions = contexts(list("abcd"), links)

    couplings = [_coupling(ctx) for ctx in regions.values()]
    assert [(coupling.group, coupling.leader) for coupling in couplings] == [
        (("a", "b"), "b"),
        (("a", "b"), "b"),
        (("c", "d"), "d"),
        (("c", "d"), "d"),
    ]


@pytest.mark.parametrize(
    ("link", "error"),
    [
        (Link("x", (("a", "p"), ("fluid", "q"))), r"couple the regions \['fluid'\]"),
        (Link("x", (("a", "p"), ("a", "q"))), "exchange 'x' joins two patches of region a"),
    ],
)
def test_region_couplings_check_the_links(link: Link, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        region_couplings(["a", "b"], [link], InProcessTransport())


def test_a_patch_is_in_one_exchange() -> None:
    links = [Link("x", (("a", "p"), ("b", "p"))), Link("y", (("a", "p"), ("b", "q")))]

    with pytest.raises(ValueError, match="a.p is in the exchanges 'x' and 'y'"):
        region_couplings(["a", "b"], links, InProcessTransport())


def test_a_region_no_exchange_names_is_not_coupled() -> None:
    couplings = region_couplings(
        ["a", "b", "c"], [Link("x", (("a", "p"), ("b", "p")))], InProcessTransport()
    )

    assert list(couplings) == ["a", "b"]


def test_run_regions_raises_the_deadlock() -> None:
    transport = InProcessTransport()

    with pytest.raises(CouplingDeadlock, match="a waits for 'x' from b"):
        run_regions(
            {
                "a": lambda: transport.receive("b", "a", "x"),
                "b": lambda: transport.receive("a", "b", "y"),
            },
            transport,
        )


def test_run_regions_raises_the_error_before_a_transport_closed_of_another_region() -> None:
    transport = InProcessTransport(timeout=TIMEOUT)

    def closed() -> None:
        raise TransportClosed("a follows")

    def deadlock() -> None:
        # returns only once a failed: b's error comes second
        with pytest.raises(TransportClosed):
            transport.receive("a", "b", "x")
        raise CouplingDeadlock("b deadlocked")

    with pytest.raises(CouplingDeadlock, match="b deadlocked"):
        run_regions({"a": closed, "b": deadlock}, transport)
