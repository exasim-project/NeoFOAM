# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun's coupling of a case and its comm channel timeout.

``cases/couplings/chain`` couples the chain a - b - c in two loops, outer to inner.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any

import pytest

from neofoam.framework.coupling import (
    AitkenRelaxation,
    CouplingConfig,
    InProcessCommChannel,
    ParticipantExchanges,
    RelativeConvergence,
)
from neofoam.framework.coupling import config as config_module
from neofoam.framework.coupling import exchanges as exchanges_module
from neofoam.framework.types import Placement

MULTI_RUN = importlib.import_module("neofoam.solver.multiRun.multiRun")
CASES = Path(__file__).parent / "cases"
CHAIN = CASES / "couplings" / "chain"
REGIONS = ["a", "b", "c"]


def _couplings(case: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """multiRun's coupling of the regions a, b and c in a copy of ``case``."""
    shutil.copytree(case, tmp_path / "case")
    monkeypatch.chdir(tmp_path / "case")
    result: dict[str, ParticipantExchanges] = MULTI_RUN.couplings(REGIONS, InProcessCommChannel())
    return result


def test_couplings_builds_a_loop_per_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    couplings = _couplings(CHAIN, tmp_path, monkeypatch)

    assert [loop.name for loop in couplings["b"].loops] == ["outer", "inner"]
    assert [loop.name for loop in couplings["a"].loops] == ["inner"]
    assert [loop.name for loop in couplings["c"].loops] == ["outer"]
    outer, inner = couplings["b"].loops
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


def test_couplings_check_the_case_s_coupling_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks: list[list[str]] = []
    check = exchanges_module.check_exchanges

    def counted(regions: list[str], *args: Any) -> None:
        checks.append(list(regions))
        check(regions, *args)

    monkeypatch.setattr(exchanges_module, "check_exchanges", counted)
    monkeypatch.setattr(config_module, "check_exchanges", counted)

    _couplings(CHAIN, tmp_path, monkeypatch)

    # once, as the file loads; its regions only are checked against the case's
    assert checks == [["a", "b", "c"]]


def test_run_hands_the_comm_channel_timeout_to_the_comm_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeouts: list[Any] = []

    class _Stop(Exception):
        pass

    def comm_channel(timeout: Any) -> None:
        timeouts.append(timeout)
        raise _Stop

    shutil.copytree(CASES.parent / "limitations" / "cases" / "pair", tmp_path / "case")
    config = CouplingConfig.load(case_dir=tmp_path / "case")
    config.commChannelTimeout = 3
    config.save(case_dir=tmp_path / "case")
    monkeypatch.chdir(tmp_path / "case")
    monkeypatch.setattr(MULTI_RUN, "InProcessCommChannel", comm_channel)

    with pytest.raises(_Stop):
        MULTI_RUN.run()

    assert timeouts == [3.0]


@pytest.mark.parametrize(("size", "path"), [("1", "threads"), ("2", "mpi")])
def test_run_takes_the_mpi_path_under_mpirun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: str, path: str
) -> None:
    taken: list[str] = []
    shutil.copytree(CASES.parents[1] / "cases" / "regionSolvers" / "entries", tmp_path / "case")
    monkeypatch.chdir(tmp_path / "case")
    monkeypatch.setattr(MULTI_RUN, "run_mpi", lambda config: taken.append("mpi") or {})
    monkeypatch.setattr(MULTI_RUN, "run_threads", lambda config: taken.append("threads") or {})
    for variable in ("PMI_SIZE", "MV2_COMM_WORLD_SIZE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", size)

    MULTI_RUN.run()

    assert taken == [path]
