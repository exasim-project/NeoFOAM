# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What each region knows of a case's coupling, the checks of exchanges and loops."""

from __future__ import annotations

import pytest
from framework.coupling.regions import contexts

from neofoam.framework.context import Context
from neofoam.framework.coupling import (
    CouplingLoop,
    Exchange,
    InProcessCommChannel,
    NearestNeighborMapping,
    ParticipantExchanges,
    coupling,
)
from neofoam.framework.coupling.exchanges import participant_exchanges
from neofoam.framework.types import Placement

X = Exchange("x", (("a", "p"), ("b", "p")))


Y = Exchange("y", (("a", "q"), ("b", "q")))


def test_a_serial_loop_enclosing_a_loop_of_two_of_its_regions_raises() -> None:
    loops = [CouplingLoop("fsi", ("x",), scheme="serial"), CouplingLoop("cht", ("y",))]

    with pytest.raises(
        ValueError,
        match=r"coupling 'fsi' is serial but encloses 'cht', which couples \['a', 'b'\] as "
        r"well; .* make 'fsi' parallel",
    ):
        participant_exchanges(["a", "b"], [X, Y], InProcessCommChannel(), loops=loops)


def test_a_serial_loop_may_enclose_a_loop_sharing_one_region() -> None:
    exchanges = [X, Exchange("z", (("b", "q"), ("c", "q")))]
    loops = [CouplingLoop("ab", ("x",), scheme="serial"), CouplingLoop("bc", ("z",))]

    couplings = participant_exchanges(
        ["a", "b", "c"], exchanges, InProcessCommChannel(), loops=loops
    )

    assert [loop.scheme for loop in couplings["b"].loops] == ["serial", "serial"]


def test_a_loop_is_serial_by_default() -> None:
    loop = CouplingLoop("n", ("x", "y"))

    assert (loop.scheme, loop.parallel, loop.placement_of("a")) == ("serial", False, Placement())


@pytest.mark.parametrize(
    ("loops", "error"),
    [
        (
            [CouplingLoop("n", ("x",)), CouplingLoop("n", ("y",))],
            "two coupling loops are named 'n'",
        ),
        (
            [CouplingLoop("n", ("x", "y")), CouplingLoop("m", ("y",))],
            "exchange 'y' is in the coupling loops 'n' and 'm'",
        ),
        ([CouplingLoop("n", ("x",))], r"the exchanges \['y'\] are in no coupling loop"),
        (
            [CouplingLoop("n", ("x", "w"))],
            "coupling loop 'n' names the exchange 'w', which does not exist",
        ),
        (
            [CouplingLoop("n", ("x", "y"), placement={"c": Placement()})],
            r"coupling loop 'n' places its op in the regions \['c'\], which it does not couple",
        ),
        (
            [
                CouplingLoop("n", ("x",), acceleration={"y": lambda s, i: s}),
                CouplingLoop("m", ("y",)),
            ],
            r"coupling loop 'n' accelerates the exchanges \['y'\], which are not its own",
        ),
    ],
    ids=[
        "duplicate loop name",
        "exchange in two loops",
        "exchange in no loop",
        "unknown exchange",
        "unknown region in placement",
        "acceleration of another exchange",
    ],
)
def test_participant_exchanges_check_the_loops(loops: list[CouplingLoop], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        participant_exchanges(["a", "b", "c"], [X, Y], InProcessCommChannel(), loops=loops)


def test_two_exchanges_of_one_name_raise() -> None:
    twin = Exchange("x", (("a", "q"), ("b", "q")))

    with pytest.raises(
        ValueError, match="two exchanges are named 'x'; an exchange's name is unique"
    ):
        participant_exchanges(
            ["a", "b"], [X, twin], InProcessCommChannel(), loops=[CouplingLoop("n", ("x",))]
        )


def test_an_exchange_maps_by_nearest_neighbor_by_default() -> None:
    assert Exchange("x", (("a", "p"), ("b", "p"))).mapping == NearestNeighborMapping()


def _coupling(ctx: Context) -> ParticipantExchanges:
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
def test_participant_exchanges_check_the_links(exchange: Exchange, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        participant_exchanges(
            ["a", "b"], [exchange], InProcessCommChannel(), [CouplingLoop("n", ("x",))]
        )


def test_a_patch_is_in_one_exchange() -> None:
    exchanges = [Exchange("x", (("a", "p"), ("b", "p"))), Exchange("y", (("a", "p"), ("b", "q")))]

    with pytest.raises(ValueError, match="a.p is in the exchanges 'x' and 'y'"):
        participant_exchanges(
            ["a", "b"], exchanges, InProcessCommChannel(), [CouplingLoop("n", ("x", "y"))]
        )


def test_a_region_no_exchange_names_is_not_coupled() -> None:
    couplings = participant_exchanges(
        ["a", "b", "c"], [X], InProcessCommChannel(), [CouplingLoop("n", ("x",))]
    )

    assert list(couplings) == ["a", "b"]
