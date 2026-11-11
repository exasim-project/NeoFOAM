# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two probe regions coupled to the fixed point of ``a = 1 + b``, ``b = 0.5 a``: (2, 1).

Serial around each ``solve_x``: b receives a's value of the same pass, so the error
halves every pass and drops below the relative limit 1e-6 after pass 20; pass 21 is the
final one, the second time step starts converged. The values are dyadic fractions,
exact in floating point, hence atol 1e-15.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from framework.coupling.regions import couple, tree

from neofoam.framework.context import Context
from neofoam.framework.coupling import CouplingNotConvergedWarning, NearestNeighborMapping
from neofoam.framework.coupling.controller import CouplingController, CouplingOutcome
from neofoam.framework.execution_observer import ExecutionObserver, observe
from neofoam.framework.initialization import InitStep
from neofoam.framework.operations import Operation, Operations
from neofoam.solver.multiRun import run
from solver.multiRun.case_files import CASES, copy_case

EXPECTED = yaml.safe_load((CASES / "affine_pair" / "expected.yaml").read_text())
PASSES = 21

Regions = dict[str, Context]


def _history(regions: Regions, region: str, time_index: int) -> list[float]:
    return [x for index, x in regions[region].models["history"] if index == time_index]


def _variant(variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(
        copy_case("affine_pair", tmp_path / "case", f"affine_pair_variants/{variant}")
    )


@pytest.fixture(scope="module")
def regions(tmp_path_factory: pytest.TempPathFactory) -> Regions:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(copy_case("affine_pair", tmp_path_factory.mktemp("affine") / "case"))
        return run()


def test_affine_pair_regions_read_their_own_config(regions: Regions) -> None:
    assert {region: ctx.models["probe"].gains for region, ctx in regions.items()} == {
        "a": {"interface.x": 1.0},
        "b": {"interface.x": 0.5},
    }
    assert [(ctx.region, ctx.time.index) for ctx in regions.values()] == [("a", 2), ("b", 2)]


def test_affine_pair_regions_couple_through_their_coupling_ops(regions: Regions) -> None:
    a_op, b_op = couple(regions["a"]), couple(regions["b"])

    assert [(op.label, op.is_leader) for op in (a_op, b_op)] == [
        ("a.coupling", False),
        ("b.coupling", True),
    ]
    assert a_op.controller is None and b_op.controller is not None
    (a_side,), (b_side,) = a_op.sides, b_op.sides
    assert [(side.end, side.field) for side in (a_side, b_side)] == [
        (("a", "interface"), "x"),
        (("b", "interface"), "x"),
    ]
    assert (a_op.partner(a_side), b_op.partner(b_side)) == ("b.interface", "a.interface")
    link = a_op.link_of(a_side)
    assert (link.name, type(link.mapping)) == ("a-b", NearestNeighborMapping)


@pytest.mark.parametrize("region", ["a", "b"])
def test_affine_pair_exchanges_the_partners_value_every_pass(regions: Regions, region: str) -> None:
    np.testing.assert_allclose(
        _history(regions, region, 1), EXPECTED[region], rtol=0, atol=1e-15, err_msg=region
    )


def test_affine_pair_second_time_step_starts_converged(regions: Regions) -> None:
    assert [len(_history(regions, region, 2)) for region in ("a", "b")] == [2, 2]


def test_affine_pair_records_the_coupling_outcome_of_every_time_step(regions: Regions) -> None:
    controller = couple(regions["b"]).controller
    assert controller is not None

    outcomes = controller.outcomes

    assert [(o.passes, o.converged) for o in outcomes] == [(PASSES, True), (2, True)]
    # a changed by 2**-20 of a_21 = 2 - 2**-20 in the final pass: 2**-21 within 1e-6
    assert outcomes[0].residuals == {"x": pytest.approx(2.0**-21, rel=1e-6)}
    assert outcomes[1].residuals["x"] < 1e-6


@pytest.mark.parametrize(
    ("variant", "passes"),
    [("serial-aitken", 5), ("parallel-constant", 40)],
)
def test_affine_pair_scheme_and_acceleration_change_only_the_passes(
    variant: str, passes: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limit 1e-6 is relative: the fixed point holds to 1e-5."""
    _variant(variant, tmp_path, monkeypatch)

    regions = run()

    final = [regions[region].fields["x"][0] for region in ("a", "b")]
    np.testing.assert_allclose(final, [2.0, 1.0], rtol=1e-5, err_msg=variant)
    assert len(_history(regions, "a", 1)) == passes


def test_affine_pair_stops_at_max_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _variant("max_iterations_5", tmp_path, monkeypatch)

    with pytest.warns(CouplingNotConvergedWarning, match="did not converge in 5 passes"):
        regions = run()

    assert _history(regions, "a", 1) == EXPECTED["a"][:5]


def test_affine_pair_a_serial_node_around_the_time_step_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _variant("no_wraps", tmp_path, monkeypatch)

    with pytest.raises(
        ValueError,
        match=r"coupling 'coupling' is serial but holds the time step of its regions; make it "
        r"parallel or place it around named operations \(wraps\.[ab]\.operations\)",
    ):
        run()


def test_affine_pair_regions_step_the_smallest_time_step_any_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _variant("a_asks_half_dt", tmp_path, monkeypatch)

    regions = run()

    assert [
        (time.value, time.index, time.delta_t)
        for time in (regions[region].time for region in ("a", "b"))
    ] == [(2.0, 4, 0.5), (2.0, 4, 0.5)]


class _Recorder(ExecutionObserver):
    """What every region reported, by region."""

    def __init__(self) -> None:
        self.graphs: dict[str, list[object]] = {}
        self.init_regions: list[set[str]] = []
        self.ran: list[tuple[str, Context]] = []
        self.tests: dict[object, list[tuple[str, int, bool]]] = {"a": [], "b": []}
        self.couplings: list[tuple[str, CouplingOutcome]] = []

    def coupling_finished(self, coupling: CouplingController, outcome: CouplingOutcome) -> None:
        self.couplings.append((coupling.label, outcome))

    def init_steps(self, steps: list[InitStep]) -> None:
        self.init_regions.append({step.region or "" for step in steps})

    def operations(self, operations: Operations, ctx: Context) -> None:
        self.graphs[ctx.region or ""] = tree(operations.ops)

    def before_operation(self, op: Operation, ctx: Context) -> None:
        self.ran.append((op.name or "", ctx))

    def after_loop_test(self, op: Operation, ctx: Context, iteration: int, result: bool) -> None:
        self.tests[ctx.region].append((op.name or "", iteration, result))


@pytest.fixture(scope="module")
def recorder(tmp_path_factory: pytest.TempPathFactory) -> tuple[_Recorder, Regions]:
    recorder = _Recorder()
    with pytest.MonkeyPatch.context() as monkeypatch, observe(recorder):
        monkeypatch.chdir(copy_case("affine_pair", tmp_path_factory.mktemp("observed") / "case"))
        return recorder, run()


def test_affine_pair_initializes_each_region_on_its_own(
    recorder: tuple[_Recorder, Regions],
) -> None:
    observed, _regions = recorder

    assert sorted(sorted(regions) for regions in observed.init_regions) == [["a"], ["b"]]


@pytest.mark.parametrize("region", ["a", "b"])
def test_affine_pair_region_graph_holds_its_solve_in_its_coupling_op(
    recorder: tuple[_Recorder, Regions], region: str
) -> None:
    observed, _regions = recorder

    assert observed.graphs[region] == [
        (
            "time_loop",
            [
                "set_time_step",
                f"{region}.time_step",
                "increment_time",
                (f"{region}.coupling", ["solve_x"]),
            ],
        )
    ]


def test_affine_pair_operations_run_on_their_region_s_context(
    recorder: tuple[_Recorder, Regions],
) -> None:
    observed, regions = recorder

    assert {(name, ctx.region) for name, ctx in observed.ran} == {
        (name, region)
        for region in ("a", "b")
        for name in ("set_time_step", f"{region}.time_step", "increment_time", "solve_x")
    }
    assert all(ctx is regions[ctx.region or ""] for _name, ctx in observed.ran)


@pytest.mark.parametrize("region", ["a", "b"])
def test_affine_pair_coupling_op_runs_the_leader_s_passes(
    recorder: tuple[_Recorder, Regions], region: str
) -> None:
    observed, _regions = recorder
    label = f"{region}.coupling"

    tests = [test for test in observed.tests[region] if test[0] == label]

    assert tests == [
        *[(label, k, True) for k in range(1, PASSES + 1)],
        (label, PASSES + 1, False),
        (label, 1, True),
        (label, 2, True),
        (label, 3, False),
    ]


def test_affine_pair_leader_reports_each_time_step_s_outcome(
    recorder: tuple[_Recorder, Regions],
) -> None:
    observed, _regions = recorder

    assert [(label, o.passes, o.converged) for label, o in observed.couplings] == [
        ("b.coupling", PASSES, True),
        ("b.coupling", 2, True),
    ]
