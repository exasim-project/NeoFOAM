# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``couplings`` of ``system/regionCoupling.yaml``: the coupling nodes multiRun builds.

``cases/couplings/chain`` couples the chain a - b - c in two nodes, outer to
inner. A broken file is that file loaded, one entry changed, and validated.
"""

import copy
import importlib
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from neofoam.framework.coupling import (
    AitkenRelaxation,
    InProcessCommChannel,
    RegionCoupling,
    RelativeConvergence,
)
from neofoam.framework.types import Placement
from neofoam.solver.multiRun.configs import RegionCouplingConfig

MULTI_RUN = importlib.import_module("neofoam.solver.multiRun.multiRun")
CASES = Path(__file__).parent / "cases"
CHAIN = CASES / "couplings" / "chain"
REGIONS = ["a", "b", "c"]


def _chain() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((CHAIN / "system" / "regionCoupling.yaml").read_text())
    return data


def _couplings(case: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """multiRun's coupling of the regions a, b and c in a copy of ``case``."""
    shutil.copytree(case, tmp_path / "case")
    monkeypatch.chdir(tmp_path / "case")
    result: dict[str, RegionCoupling] = MULTI_RUN.couplings(REGIONS, InProcessCommChannel())
    return result


def test_couplings_builds_a_node_per_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    couplings = _couplings(CHAIN, tmp_path, monkeypatch)

    assert [node.name for node in couplings["b"].nodes] == ["outer", "inner"]
    assert [node.name for node in couplings["a"].nodes] == ["inner"]
    assert [node.name for node in couplings["c"].nodes] == ["outer"]
    outer, inner = couplings["b"].nodes
    assert (outer.exchanges, outer.max_iterations, outer.scheme) == (("b-c",), 7, "parallel")
    assert (inner.exchanges, inner.max_iterations) == (("a-b",), 100)
    assert isinstance(outer.acceleration["b-c"], AitkenRelaxation)
    assert inner.acceleration == {}
    assert [(type(c), c.data, c.limit) for c in outer.convergence] == [
        (RelativeConvergence, "x", 1e-6),
        (RelativeConvergence, "y", 1e-3),
    ]
    assert inner.placement == {"b": Placement(("solve_x",))}
    assert inner.placement_of("a") == Placement()
    assert outer.placement_of("b") == Placement()


def test_couplings_gives_every_node_an_acceleration_of_its_own() -> None:
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


def test_couplings_a_serial_node_may_enclose_a_chained_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            _broken("exchanges.1", {"name": "a-b", "sides": {"b": "to_c", "c": "to_b"}}),
            "two exchanges are named 'a-b'",
        ),
        (
            _broken("exchanges.1", {"name": "b-c", "sides": {"a": "to_b", "c": "to_b"}}),
            r"a.to_b is in the exchanges 'a-b' and 'b-c'; a coupled patch is in one exchange",
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
def test_couplings_rejects_a_broken_file(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RegionCouplingConfig.model_validate(data)


def test_couplings_of_a_region_the_case_does_not_run_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(CHAIN, tmp_path / "case")
    monkeypatch.chdir(tmp_path / "case")

    with pytest.raises(ValueError, match=r"couple the regions \['c'\], which are not in"):
        MULTI_RUN.couplings(["a", "b"], InProcessCommChannel())


def _timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: Any = "none", env: Any = None
) -> Any:
    """multiRun's comm channel timeout, the chain's ``transportTimeout`` set to ``seconds``.

    ``"none"``: no case file; ``"default"``: the file as it is.
    """
    (tmp_path / "system").mkdir()
    if seconds != "none":
        config = RegionCouplingConfig.load(case_dir=CHAIN)
        if seconds != "default":
            config.transportTimeout = seconds
        config.save(case_dir=tmp_path)
    monkeypatch.chdir(tmp_path)
    if env is None:
        monkeypatch.delenv(MULTI_RUN.TIMEOUT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(MULTI_RUN.TIMEOUT_VARIABLE, env)
    return MULTI_RUN.transport_timeout()


@pytest.mark.parametrize(
    ("seconds", "timeout"),
    [("none", 600.0), ("default", 600.0), (5, 5.0), (0, None), (None, None)],
    ids=["no-file", "default", "seconds", "zero", "null"],
)
def test_the_transport_timeout_is_the_file_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: Any, timeout: Any
) -> None:
    assert _timeout(tmp_path, monkeypatch, seconds) == timeout


@pytest.mark.parametrize(("env", "timeout"), [("2.5", 2.5), ("0", None)])
def test_the_environment_variable_overrides_the_file_s_transport_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: str, timeout: Any
) -> None:
    assert _timeout(tmp_path, monkeypatch, 5, env) == timeout


@pytest.mark.parametrize("env", ["soon", "-1", "inf"])
def test_a_broken_environment_variable_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: str
) -> None:
    with pytest.raises(ValueError, match="NEOFOAM_COUPLING_TIMEOUT=.*at least 0"):
        _timeout(tmp_path, monkeypatch, env=env)


@pytest.mark.parametrize("seconds", [-1, float("inf")])
def test_a_broken_transport_timeout_raises(seconds: float) -> None:
    with pytest.raises(ValidationError, match="transportTimeout"):
        RegionCouplingConfig.model_validate({**_chain(), "transportTimeout": seconds})


def test_run_hands_the_transport_timeout_to_the_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeouts: list[Any] = []

    class _Stop(Exception):
        pass

    def comm_channel(timeout: Any) -> None:
        timeouts.append(timeout)
        raise _Stop

    shutil.copytree(CASES.parent / "limitations" / "cases" / "pair", tmp_path / "case")
    monkeypatch.chdir(tmp_path / "case")
    monkeypatch.setenv(MULTI_RUN.TIMEOUT_VARIABLE, "3")
    monkeypatch.setattr(MULTI_RUN, "InProcessCommChannel", comm_channel)

    with pytest.raises(_Stop):
        MULTI_RUN.run()

    assert timeouts == [3.0]
