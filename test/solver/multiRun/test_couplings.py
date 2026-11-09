# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``couplings`` of ``system/regionCoupling.yaml``: the coupling nodes multiRun builds.

The regions ``a``, ``b`` and ``c`` form a chain: the exchange ``a-b`` joins
a's ``to_b`` with b's ``to_a``, ``b-c`` b's ``to_c`` with c's ``to_b``. With
``couplings`` the file lists the nodes outer to inner; without it one node
``coupling`` holds every exchange. Every broken rule fails as the file loads.
"""

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from neofoam.framework.coupling import (
    AitkenRelaxation,
    ConstantRelaxation,
    InProcessCommChannel,
    RegionCoupling,
    RelativeConvergence,
)
from neofoam.framework.types import Placement
from neofoam.solver.multiRun.configs import RegionCouplingConfig

MULTI_RUN = importlib.import_module("neofoam.solver.multiRun.multiRun")
REGIONS = ["a", "b", "c"]
EXCHANGES = [
    {"sides": {"a": "to_b", "b": "to_a"}},
    {"sides": {"b": "to_c", "c": "to_b"}},
]
NODES: dict[str, Any] = {
    "exchanges": EXCHANGES,
    "scheme": "parallel",
    "couplings": [
        {
            "name": "outer",
            "maxIterations": 7,
            "exchanges": [{"name": "b-c", "acceleration": {"type": "aitken", "relaxation": 0.5}}],
            "convergence": [
                {"type": "relative", "data": "x", "limit": 1e-6},
                {"type": "relative", "data": "y", "limit": 1e-3},
            ],
        },
        {
            "name": "inner",
            "exchanges": ["a-b"],
            "convergence": [{"type": "relative", "data": "x", "limit": 1e-8}],
            "wraps": {"b": {"operations": ["solve_x"], "downstream": False}},
        },
    ],
}


def _couplings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: Any
) -> dict[str, RegionCoupling]:
    """multiRun's coupling of the regions as ``data`` written to the case file describes."""
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "regionCoupling.yaml").write_text(yaml.safe_dump(data))
    monkeypatch.chdir(tmp_path)
    result: dict[str, RegionCoupling] = MULTI_RUN.couplings(REGIONS, InProcessCommChannel())
    return result


def test_couplings_reads_the_nodes_outer_to_inner() -> None:
    config = RegionCouplingConfig.model_validate(NODES)

    outer, inner = config.nodes()
    assert [outer.name, inner.name] == ["outer", "inner"]
    assert [(e.name, e.acceleration) for e in inner.exchanges] == [("a-b", None)]
    assert outer.exchanges[0].acceleration is not None
    assert outer.exchanges[0].acceleration.type == "aitken"
    assert [(c.data, c.limit) for c in outer.convergence] == [("x", 1e-6), ("y", 1e-3)]
    assert (outer.maxIterations, inner.maxIterations) == (7, 100)
    assert outer.wraps == {}
    assert inner.wraps["b"].model_dump() == {"operations": ["solve_x"], "downstream": False}


def test_couplings_builds_a_node_per_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    couplings = _couplings(tmp_path, monkeypatch, NODES)

    assert [node.name for node in couplings["b"].nodes] == ["outer", "inner"]
    assert [node.name for node in couplings["a"].nodes] == ["inner"]
    assert [node.name for node in couplings["c"].nodes] == ["outer"]
    assert [node.scheme for node in couplings["b"].nodes] == ["parallel", "parallel"]
    outer, inner = couplings["b"].nodes
    assert (outer.exchanges, outer.max_iterations) == (("b-c",), 7)
    assert isinstance(outer.acceleration["b-c"], AitkenRelaxation)
    assert inner.acceleration == {}
    assert [(type(c), c.data, c.limit) for c in outer.convergence] == [  # type: ignore[attr-defined]
        (RelativeConvergence, "x", 1e-6),
        (RelativeConvergence, "y", 1e-3),
    ]
    assert inner.placement == {"b": Placement(("solve_x",), downstream=False)}
    assert inner.placement_of("a") == Placement()
    assert outer.placement_of("b") == Placement()


def test_couplings_without_nodes_puts_every_exchange_in_one_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {
        "exchanges": EXCHANGES,
        "acceleration": {"type": "constant", "relaxation": 0.7},
        "convergence": [{"type": "relative", "data": "x", "limit": 1e-6}],
        "maxIterations": 9,
    }

    couplings = _couplings(tmp_path, monkeypatch, data)

    (node,) = couplings["a"].nodes
    assert couplings["b"].nodes == couplings["c"].nodes == (node,)
    assert (node.name, node.exchanges, node.max_iterations) == ("coupling", ("a-b", "b-c"), 9)
    assert sorted(node.acceleration) == ["a-b", "b-c"]
    assert all(type(a) is ConstantRelaxation and a.omega == 0.7 for a in node.acceleration.values())
    assert node.acceleration["a-b"] is not node.acceleration["b-c"]
    assert node.placement == {}


def test_couplings_without_nodes_and_acceleration_relaxes_nothing() -> None:
    data = {
        "exchanges": EXCHANGES,
        "convergence": [{"type": "relative", "data": "x", "limit": 1e-6}],
    }

    (node,) = RegionCouplingConfig.model_validate(data).nodes()

    assert (node.name, node.maxIterations) == ("coupling", 100)
    assert [(e.name, e.acceleration) for e in node.exchanges] == [("a-b", None), ("b-c", None)]


def test_couplings_takes_the_scheme_per_node_by_default_the_file_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _broken("scheme", "serial")
    data["couplings"][0]["scheme"] = "parallel"

    couplings = _couplings(tmp_path, monkeypatch, data)

    assert [(node.name, node.scheme) for node in couplings["b"].nodes] == [
        ("outer", "parallel"),
        ("inner", "serial"),
    ]


def test_couplings_are_parallel_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "exchanges": EXCHANGES,
        "convergence": [{"type": "relative", "data": "x", "limit": 1e-6}],
    }

    couplings = _couplings(tmp_path, monkeypatch, data)

    assert RegionCouplingConfig.model_validate(data).scheme == "parallel"
    assert [node.scheme for node in couplings["a"].nodes] == ["parallel"]
    nodes = {key: value for key, value in NODES.items() if key != "scheme"}
    assert [node.scheme for node in RegionCouplingConfig.model_validate(nodes).nodes()] == [
        "parallel",
        "parallel",
    ]


def test_couplings_a_serial_node_may_enclose_a_chained_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # outer couples b and c, inner a and b: they share b only
    couplings = _couplings(tmp_path, monkeypatch, _broken("couplings.0.scheme", "serial"))

    assert [node.scheme for node in couplings["b"].nodes] == ["serial", "parallel"]


def test_couplings_a_serial_node_enclosing_a_node_of_two_of_its_regions_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {
        "exchanges": [
            {"name": "fsi-ab", "sides": {"a": "to_b", "b": "to_a"}},
            {"name": "cht-ab", "sides": {"a": "wall", "b": "wall"}},
        ],
        "couplings": [
            {
                "name": "fsi",
                "scheme": "serial",
                "exchanges": ["fsi-ab"],
                "convergence": [{"type": "relative", "data": "x", "limit": 1e-6}],
            },
            {
                "name": "cht",
                "exchanges": ["cht-ab"],
                "convergence": [{"type": "relative", "data": "y", "limit": 1e-6}],
            },
        ],
    }

    with pytest.raises(
        ValueError,
        match=r"coupling 'fsi' is serial but encloses 'cht', which couples \['a', 'b'\] as "
        r"well; .* make 'fsi' parallel",
    ):
        _couplings(tmp_path, monkeypatch, data)


def _broken(path: str, value: Any) -> dict[str, Any]:
    """``NODES`` with the entry at ``path`` (``.``-separated keys and indices) set to ``value``."""
    data = copy.deepcopy(NODES)
    *parents, last = path.split(".")
    target: Any = data
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_broken("couplings.1.name", "outer"), "two coupling nodes are named 'outer'"),
        (
            _broken("couplings.1.exchanges", ["a-b", "b-c"]),
            "exchange 'b-c' is in the nodes 'outer' and 'inner'",
        ),
        (_broken("couplings.1.exchanges", ["b-c-2"]), "names the exchange 'b-c-2', which does not"),
        (
            {**NODES, "couplings": NODES["couplings"][:1]},
            r"the exchanges \['a-b'\] are in no coupling node",
        ),
        (_broken("couplings.0.maxIteration", 5), r"couplings.0.maxIteration\n.*Extra inputs"),
        (
            _broken("couplings.0.exchanges.0.acceleration", {"type": "iqn-ils", "relaxation": 0.5}),
            r"couplings.0.exchanges.0.acceleration.type\n.*unknown acceleration 'iqn-ils'",
        ),
        (
            _broken("couplings.1.wraps.b.operation", ["solve_x"]),
            r"couplings.1.wraps.b.operation\n.*Extra inputs",
        ),
        (
            _broken("couplings.1.wraps", {"c": {"operations": ["solve_x"]}}),
            r"node 'inner' wraps the regions \['c'\], which it does not couple",
        ),
        (
            {**NODES, "maxIterations": 5},
            r"with couplings the nodes set \['maxIterations'\]; remove the top-level",
        ),
        (
            {**NODES, "convergence": [{"type": "relative", "data": "x", "limit": 1e-6}]},
            r"remove the top-level \['convergence'\]",
        ),
        ({**NODES, "coupling": []}, r"coupling\n.*Extra inputs"),
    ],
    ids=[
        "duplicate-name",
        "exchange-in-two-nodes",
        "unknown-exchange",
        "exchange-in-no-node",
        "typo-in-a-node",
        "unknown-acceleration",
        "typo-in-wraps",
        "wraps-a-region-of-another-node",
        "top-level-maxIterations",
        "top-level-convergence",
        "typo-at-the-top",
    ],
)
def test_couplings_rejects_a_broken_file(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RegionCouplingConfig.model_validate(data)


def test_couplings_rejects_a_broken_file_as_it_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValidationError, match="two coupling nodes are named 'outer'"):
        _couplings(tmp_path, monkeypatch, _broken("couplings.1.name", "outer"))
