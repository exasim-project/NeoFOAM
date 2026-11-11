# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun's coupling of a case and its comm channel timeout.

``cases/couplings/chain`` couples the chain a - b - c in two nodes, outer to inner.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any

import pytest

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


def test_couplings_of_a_region_the_case_does_not_run_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(CHAIN, tmp_path / "case")
    monkeypatch.chdir(tmp_path / "case")

    with pytest.raises(ValueError, match=r"couple the regions \['c'\], which are not in"):
        MULTI_RUN.couplings(["a", "b"], InProcessCommChannel())


def _timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: Any) -> Any:
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
    config = RegionCouplingConfig.load(case_dir=tmp_path / "case")
    config.transportTimeout = 3
    config.save(case_dir=tmp_path / "case")
    monkeypatch.chdir(tmp_path / "case")
    monkeypatch.setattr(MULTI_RUN, "InProcessCommChannel", comm_channel)

    with pytest.raises(_Stop):
        MULTI_RUN.run()

    assert timeouts == [3.0]
