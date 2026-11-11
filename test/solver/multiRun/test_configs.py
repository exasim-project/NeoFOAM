# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``system/regionCoupling.yaml`` read into :class:`RegionCouplingConfig`.

``cases/region_coupling/minimal`` sets only the required entries, every other
``cases/region_coupling/<name>`` breaks one rule; a broken chain is
``cases/couplings/chain`` with one entry changed.
"""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from neofoam.framework.coupling import NearestNeighborMapping
from neofoam.solver.multiRun.configs import RegionCouplingConfig

CASES = Path(__file__).parent / "cases"
CHAIN = CASES / "couplings" / "chain"


def _chain() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((CHAIN / "system" / "regionCoupling.yaml").read_text())
    return data


def test_region_coupling_config_reads_every_entry() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES / "coupled_slabs")

    assert config.model_dump() == {
        "exchanges": [
            {
                "name": "interface",
                "ports": {"left": "right", "right": "left"},
                "mapping": {"type": "nearest-neighbor"},
            }
        ],
        "couplings": [
            {
                "name": "coupling",
                "scheme": "parallel",
                "maxIterations": 100,
                "exchanges": [
                    {"name": "interface", "acceleration": {"type": "aitken", "relaxation": 0.5}}
                ],
                "convergence": [
                    {"data": "T_sink", "type": "relative", "limit": 1e-12},
                    {"data": "h", "type": "relative", "limit": 1e-12},
                ],
                "wraps": {},
            }
        ],
        "commChannelTimeout": 600.0,
    }


def test_region_coupling_config_defaults_the_optional_entries() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES / "region_coupling" / "minimal")

    assert config.model_dump() == {
        "exchanges": [
            {
                "name": None,
                "ports": {"solid": "interface", "fluid": "interface"},
                "mapping": {"type": "nearest-neighbor"},
            }
        ],
        "couplings": [
            {
                "name": "thermal",
                "scheme": "serial",
                "maxIterations": 100,
                "exchanges": [{"name": "solid-fluid", "acceleration": None}],
                "convergence": [{"data": "T", "type": "relative", "limit": 1e-5}],
                "wraps": {},
            }
        ],
        "commChannelTimeout": 600.0,
    }


def test_region_coupling_config_reads_a_list_of_exchanges() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES.parent / "limitations" / "cases" / "chain")

    assert [(e.name, e.ports, type(e.mapping)) for e in config.exchanges] == [
        (None, {"a": "to_b", "b": "to_a"}, NearestNeighborMapping),
        (None, {"b": "to_c", "c": "to_b"}, NearestNeighborMapping),
    ]


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("unknown_scheme", r"couplings.0.scheme\n.*Input should be 'serial' or 'parallel'"),
        (
            "unknown_mapping",
            r"exchanges.0.mapping.mapping\n.*Input tag 'rbf' .* expected tags: 'identity', "
            r"'nearest-neighbor'",
        ),
        (
            "unknown_acceleration",
            r"acceleration.acceleration\n.*Input tag 'anderson' .* expected tags: 'constant', "
            r"'aitken'",
        ),
        (
            "acceleration_option_typo",
            r"acceleration.aitken.relaxation\n.*Field required.*\n.*\n.*acceleration.aitken."
            r"relax\n.*Extra inputs",
        ),
        (
            "relaxation_above_one",
            r"acceleration.constant.relaxation\n.*less than or equal to 1",
        ),
        ("unknown_convergence", r"convergence.0.convergence\n.*Input tag 'absolute' .* 'relative'"),
        (
            "convergence_option_typo",
            r"convergence.relative.limit\n.*Field required.*\n.*\n.*convergence.relative."
            r"lmit\n.*Extra inputs",
        ),
        ("no_convergence", r"couplings.0.convergence\n.*at least 1 item"),
        ("three_regions", r"exchanges.0.ports\n.*at most 2 items"),
        ("top_level_convergence", r"couplings\n.*Field required.*\n.*\nconvergence\n.*Extra"),
        ("sides", r"exchanges.0.ports\n.*Field required.*\n.*\nexchanges.0.sides\n.*Extra"),
    ],
)
def test_region_coupling_config_rejects_an_invalid_file(case: str, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        RegionCouplingConfig.load(case_dir=CASES / "region_coupling" / case)


def test_region_coupling_config_gives_every_node_an_acceleration_of_its_own() -> None:
    config = RegionCouplingConfig.load(case_dir=CHAIN)

    first, second = config.nodes()[0], config.nodes()[0]

    assert first.acceleration["b-c"] is not second.acceleration["b-c"]


def test_a_node_without_a_scheme_is_serial() -> None:
    data = _chain()
    del data["couplings"][1]["scheme"]

    nodes = RegionCouplingConfig.model_validate(data).nodes()

    assert [(node.name, node.scheme) for node in nodes] == [
        ("outer", "parallel"),
        ("inner", "serial"),
    ]


def test_a_serial_node_may_enclose_a_chained_node() -> None:
    """outer couples b and c, inner a and b: they share b only."""
    data = _chain()
    data["couplings"][0]["scheme"] = "serial"

    nodes = RegionCouplingConfig.model_validate(data).nodes()

    assert [node.scheme for node in nodes] == ["serial", "parallel"]


def test_a_serial_node_enclosing_a_node_of_two_of_its_regions_raises_as_the_file_loads() -> None:
    with pytest.raises(
        ValidationError,
        match=r"coupling 'fsi' is serial but encloses 'cht', which couples \['a', 'b'\] as "
        r"well; .* make 'fsi' parallel",
    ):
        RegionCouplingConfig.load(case_dir=CASES / "couplings" / "serial_encloses")


def _broken(path: str, value: Any) -> dict[str, Any]:
    """The chain with the entry at ``path`` (``.``-separated keys and indices) set to ``value``."""
    data = copy.deepcopy(_chain())
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
            "exchange 'b-c' is in the coupling nodes 'outer' and 'inner'",
        ),
        (_broken("couplings.1.exchanges", ["b-c-2"]), "names the exchange 'b-c-2', which does not"),
        (
            _broken("couplings", _chain()["couplings"][:1]),
            r"the exchanges \['a-b'\] are in no coupling node",
        ),
        (
            _broken("exchanges.1", {"name": "a-b", "ports": {"b": "to_c", "c": "to_b"}}),
            "two exchanges are named 'a-b'",
        ),
        (
            _broken("exchanges.1", {"name": "b-c", "ports": {"a": "to_b", "c": "to_b"}}),
            r"a.to_b is in the exchanges 'a-b' and 'b-c'; a coupled port is in one exchange",
        ),
        (_broken("couplings.0.maxIteration", 5), r"couplings.0.maxIteration\n.*Extra inputs"),
        (
            _broken("couplings.0.exchanges.0.acceleration", {"type": "iqn-ils", "relaxation": 0.5}),
            r"couplings.0.exchanges.0.acceleration.acceleration\n.*Input tag 'iqn-ils' .* "
            r"expected tags: 'constant', 'aitken'",
        ),
        (
            _broken("couplings.1.wraps.b.operation", ["solve_x"]),
            r"couplings.1.wraps.b.operation\n.*Extra inputs",
        ),
        (
            _broken("couplings.1.wraps.b.downstream", False),
            r"couplings.1.wraps.b.downstream\n.*Extra inputs",
        ),
        (
            _broken("couplings.1.wraps", {"c": {"operations": ["solve_x"]}}),
            r"node 'inner' places its op in the regions \['c'\], which it does not couple",
        ),
        (_broken("maxIterations", 5), r"maxIterations\n.*Extra inputs"),
        (_broken("coupling", []), r"coupling\n.*Extra inputs"),
    ],
    ids=[
        "duplicate-node-name",
        "exchange-in-two-nodes",
        "unknown-exchange",
        "exchange-in-no-node",
        "duplicate-exchange-name",
        "patch-in-two-exchanges",
        "typo-in-a-node",
        "unknown-acceleration",
        "typo-in-wraps",
        "downstream-in-wraps",
        "wraps-a-region-of-another-node",
        "top-level-maxIterations",
        "typo-at-the-top",
    ],
)
def test_region_coupling_config_rejects_a_broken_chain(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RegionCouplingConfig.model_validate(data)


@pytest.mark.parametrize("seconds", [-1, float("inf")])
def test_region_coupling_config_rejects_a_broken_comm_channel_timeout(seconds: float) -> None:
    with pytest.raises(ValidationError, match="commChannelTimeout"):
        RegionCouplingConfig.model_validate({**_chain(), "commChannelTimeout": seconds})
