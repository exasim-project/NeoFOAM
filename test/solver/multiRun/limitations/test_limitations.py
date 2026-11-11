# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What the coupling design does with each case of ``cases/``.

Each case is a multiRun case of the ``probe`` solver, coupled parallel. A test
marked ``xfail(strict=True)`` asserts the behaviour we want and fails today with
the error it names; the test next to it pins what happens today. The limit 1e-6
is relative: the fixed points hold to 1e-5.
"""

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from framework.coupling.regions import couple

from neofoam.framework.context import Context
from neofoam.solver.multiRun import run

HERE = Path(__file__).parent

Regions = dict[str, Context]


def _case(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: Optional[str] = None
) -> None:
    shutil.copytree(HERE / "cases" / case, tmp_path / "case")
    if variant is not None:
        shutil.copytree(HERE / "variants" / variant, tmp_path / "case", dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path / "case")


def _assert_final(regions: Regions, expected: dict[str, float], case: str) -> None:
    final = {region: ctx.models["history"][-1][1] for region, ctx in regions.items()}
    assert list(final) == list(expected), case
    np.testing.assert_allclose(
        list(final.values()), list(expected.values()), rtol=1e-5, err_msg=case
    )


def _solves(regions: Regions) -> dict[str, int]:
    return {region: len(ctx.models["history"]) for region, ctx in regions.items()}


def test_pair_reaches_the_fixed_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _case("pair", tmp_path, monkeypatch)

    regions = run()

    _assert_final(regions, {"a": 2.0, "b": 1.0}, "pair")
    # each side receives what the other sent the pass before: 39 passes, then the final one
    assert _solves(regions) == {"a": 40, "b": 40}


def test_chain_reaches_the_fixed_point_in_one_coupling_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both exchanges share b: one loop over a, b and c, led by c."""
    _case("chain", tmp_path, monkeypatch)

    regions = run()

    _assert_final(regions, {"a": 3.0, "b": 2.0, "c": 2.0}, "chain")
    assert _solves(regions) == {"a": 58, "b": 58, "c": 58}
    assert [
        (op.label, [(s.patch, op.exchange_of(s).name) for s in op.sides], op.is_leader)
        for op in (couple(ctx) for ctx in regions.values())
    ] == [
        ("a.coupling", [("to_b", "a-b")], False),
        ("b.coupling", [("to_a", "a-b"), ("to_c", "b-c")], False),
        ("c.coupling", [("to_b", "b-c")], True),
    ]


def test_two_interfaces_reach_the_fixed_point_in_one_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case("two_interfaces", tmp_path, monkeypatch)

    regions = run()

    _assert_final(regions, {"a": 2.0, "b": 1.0}, "two_interfaces")
    assert _solves(regions) == {"a": 40, "b": 40}
    a = couple(regions["a"])
    assert [a.exchange_of(side).name for side in a.sides] == ["a-b", "a-b-2"]


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="the case maps nearest-neighbor (the default), which needs coordinates",
)
def test_no_coordinates_reaches_the_fixed_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case("no_coordinates", tmp_path, monkeypatch)

    regions = run()

    _assert_final(regions, {"a": 2.0, "b": 1.0}, "no_coordinates")


def test_no_coordinates_fails_when_the_exchange_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case("no_coordinates", tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="'nearest-neighbor' needs the coordinates of both sides"):
        run()


def test_no_coordinates_with_the_identity_mapping_reaches_the_fixed_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case("no_coordinates", tmp_path, monkeypatch, "no_coordinates_identity")

    regions = run()

    _assert_final(regions, {"a": 2.0, "b": 1.0}, "no_coordinates identity")


def test_renamed_fields_reach_the_fixed_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One model per direction: a sends X and receives Y, b the other way round."""
    _case("renamed_fields", tmp_path, monkeypatch)

    regions = run()

    _assert_final(regions, {"a": 2.0, "b": 1.0}, "renamed_fields")
    sides = [side for ctx in regions.values() for side in couple(ctx).sides]
    assert [(s.sends.__name__, s.receives.__name__) for s in sides] == [("X", "Y"), ("Y", "X")]


def test_convergence_on_unknown_data_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``data: T`` matches no side (both solve x): the leader raises building its controller."""
    _case("convergence_typo", tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match=r"convergence RelativeConvergence on T, which no side of the node sends"
    ):
        run()
