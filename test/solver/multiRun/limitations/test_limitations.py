# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What the coupling design does with each case of ``cases/`` (see README.md).

Each case is a normal multiRun case of the ``probe`` solver. A test marked
``xfail(strict=True)`` asserts the behaviour we want and fails today with the
error it names; the test next to it pins what happens today instead.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any

import pytest

from neofoam.framework.context import Context
from neofoam.framework.coupling import Coupling, CouplingOp, IdentityMapping
from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun.configs import RegionCouplingConfig
from solver.multiRun.limitations.probe_solver import probe

CASES = Path(__file__).parent / "cases"
MULTI_RUN = importlib.import_module("neofoam.solver.multiRun.multiRun")


Regions = dict[str, Context]


def _run(case: Any, monkeypatch: pytest.MonkeyPatch) -> Regions:
    """multiRun in ``cases/<case>`` (or the directory ``case``) with the probe solver."""
    monkeypatch.setattr(
        MULTI_RUN, "resolve_solver", lambda name: probe if name == "probe" else resolve_solver(name)
    )
    monkeypatch.chdir(case if isinstance(case, Path) else CASES / case)
    regions: Regions = MULTI_RUN.run()
    return regions


def _final(regions: Regions) -> dict[str, float]:
    """The last value each region solved."""
    return {region: ctx.models["history"][-1] for region, ctx in regions.items()}


def _solves(regions: Regions) -> dict[str, int]:
    return {region: len(ctx.models["history"]) for region, ctx in regions.items()}


def _coupling(ctx: Context) -> CouplingOp:
    op = Coupling.of(ctx).operations()[0].func
    assert isinstance(op, CouplingOp)
    return op


# --- pair: the baseline that works -------------------------------------------------------------


def test_pair_reaches_the_fixed_point(monkeypatch: pytest.MonkeyPatch) -> None:
    regions = _run("pair", monkeypatch)

    assert _final(regions) == pytest.approx({"a": 2.0, "b": 1.0}, rel=1e-5)
    # parallel (the default): each side receives what the other sent the pass before;
    # 39 passes to converge, then the final pass
    assert _solves(regions) == {"a": 40, "b": 40}


# --- chain a <-> b <-> c: two exchanges -------------------------------------------------------


def test_chain_reaches_the_fixed_point_in_one_coupling_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both exchanges share region b: one group over a, b and c (block Jacobi: parallel, the
    default), led by c, not two nested loops (which took a and b 122 solves, c 11)."""
    regions = _run("chain", monkeypatch)

    assert _final(regions) == pytest.approx({"a": 3.0, "b": 2.0, "c": 2.0}, rel=1e-5)
    # 57 passes to converge, then the final pass
    assert _solves(regions) == {"a": 58, "b": 58, "c": 58}
    couplings = {region: _coupling(ctx) for region, ctx in regions.items()}
    # b's coupling op hands both its sides their data
    assert [
        (op.label, [(s.patch, op.exchange_of(s).name) for s in op.sides], op.is_leader)
        for op in couplings.values()
    ] == [
        ("a.coupling", [("to_b", "a-b")], False),
        ("b.coupling", [("to_a", "a-b"), ("to_c", "b-c")], False),
        ("c.coupling", [("to_b", "b-c")], True),
    ]


# --- two exchanges between the same pair of regions -------------------------------------------


def test_two_interfaces_reach_the_fixed_point_in_one_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    regions = _run("two_interfaces", monkeypatch)

    assert _final(regions) == pytest.approx({"a": 2.0, "b": 1.0}, rel=1e-5)
    # one loop of both exchanges (nested loops took 113): 39 passes, then the final pass
    assert _solves(regions) == {"a": 40, "b": 40}
    a = _coupling(regions["a"])
    assert [a.exchange_of(side).name for side in a.sides] == ["a-b", "a-b-2"]


# --- a scalar exchange without coordinates ----------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="the case maps nearest-neighbor (the default), which needs coordinates",
)
def test_no_coordinates_reaches_the_fixed_point(monkeypatch: pytest.MonkeyPatch) -> None:
    regions = _run("no_coordinates", monkeypatch)

    assert _final(regions) == pytest.approx({"a": 2.0, "b": 1.0}, rel=1e-5)


def test_no_coordinates_fails_when_the_exchange_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="'nearest-neighbor' needs the coordinates of both sides"):
        _run("no_coordinates", monkeypatch)


def test_no_coordinates_with_the_identity_mapping_reaches_the_fixed_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = tmp_path / "no_coordinates"
    shutil.copytree(CASES / "no_coordinates", case)
    config = RegionCouplingConfig.load(case_dir=case)
    config.exchanges[0].mapping = IdentityMapping()
    config.save(case_dir=case)

    regions = _run(case, monkeypatch)

    assert _final(regions) == pytest.approx({"a": 2.0, "b": 1.0}, rel=1e-5)


# --- different field names on the two sides (x vs y), one quantity each way -------------------


def test_renamed_fields_reach_the_fixed_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """One model per direction: a sends X and receives Y, b sends Y and receives X."""
    regions = _run("renamed_fields", monkeypatch)

    assert _final(regions) == pytest.approx({"a": 2.0, "b": 1.0}, rel=1e-5)
    sides = [side for ctx in regions.values() for side in _coupling(ctx).sides]
    assert [(s.sends.__name__, s.receives.__name__) for s in sides] == [("X", "Y"), ("Y", "X")]


# --- a convergence measure naming data nobody exchanges ---------------------------------------


def test_convergence_on_unknown_data_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``data: T`` matches no side (both solve x): the leader raises as it builds its controller."""
    with pytest.raises(
        ValueError, match=r"convergence RelativeConvergence on T, which no side of the node sends"
    ):
        _run("convergence_typo", monkeypatch)
