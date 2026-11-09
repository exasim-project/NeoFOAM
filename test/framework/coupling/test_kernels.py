# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The numpy kernels of a coupling, each on a few hand-checked values.

The relaxations are fed the fixed-point iteration ``x -> 0.5 * x + 1`` (fixed
point 2), the way a coupling feeds them: what the other side sends back for the
value last delivered. Aitken's factor is exact for a linear map of one value, so
its second relaxed pass lands on the fixed point (to round-off, hence 1e-12).
"""

import tracemalloc
from typing import Any

import numpy as np
import pytest

from neofoam.framework.coupling import (
    AitkenRelaxation,
    ConstantRelaxation,
    IdentityMapping,
    Mapping,
    NearestNeighborMapping,
    RelativeConvergence,
    SideChange,
    kernels,
    nearest_neighbor,
    strategy,
)


def _points(coordinates: Any) -> Any:
    return None if coordinates is None else np.array(coordinates)


def _passes(relaxation: ConstantRelaxation, count: int) -> list[float]:
    """The values delivered in ``count`` passes of one time step, starting from 0."""
    delivered = [float(relaxation([{"x": np.zeros(1)}], 1)[0]["x"][0])]
    for iteration in range(2, count + 1):
        sent_back = 0.5 * delivered[-1] + 1.0
        delivered.append(float(relaxation([{"x": np.full(1, sent_back)}], iteration)[0]["x"][0]))
    return delivered


def test_nearest_neighbor_picks_the_closest_source_point() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    target = np.array([[1.9, 0.1], [0.2, 0.0], [1.1, -0.3]])

    assert nearest_neighbor(source, target).tolist() == [2, 0, 1]


def test_constant_relaxation_moves_by_the_factor() -> None:
    delivered = _passes(ConstantRelaxation(relaxation=0.5), 3)

    np.testing.assert_allclose(delivered, [0.0, 0.5, 0.875], rtol=1e-15, err_msg="omega 0.5")


def test_aitken_relaxation_reaches_the_fixed_point_of_a_linear_map() -> None:
    delivered = _passes(AitkenRelaxation(relaxation=0.5), 3)

    np.testing.assert_allclose(delivered, [0.0, 0.5, 2.0], rtol=1e-12, err_msg="Aitken")


def test_aitken_relaxation_restarts_from_initial_each_time_step() -> None:
    relaxation = AitkenRelaxation(relaxation=0.5)
    _passes(relaxation, 3)

    first_pass = relaxation([{"x": np.full(1, 4.0)}], 1)[0]["x"]

    np.testing.assert_allclose(first_pass, [3.0], rtol=1e-12, err_msg="2 + 0.5 * (4 - 2)")


def test_aitken_relaxation_restarts_after_a_round_off_residual() -> None:
    relaxation = AitkenRelaxation(relaxation=0.5)
    relaxation([{"x": np.full(1, 2.0)}], 1)
    relaxation([{"x": np.full(1, 2.0)}], 2)

    relaxed = relaxation([{"x": np.full(1, 4.0)}], 3)[0]["x"]

    np.testing.assert_allclose(relaxed, [3.0], rtol=1e-12, err_msg="2 + 0.5 * (4 - 2)")


def test_relaxation_keeps_the_names_and_shapes_of_the_data() -> None:
    sent = [{"T_sink": np.zeros(2), "h": np.ones(2)}, {"T_sink": np.full(2, 3.0)}]

    relaxed = ConstantRelaxation(relaxation=0.5)(sent, 1)

    assert [{name: values.tolist() for name, values in data.items()} for data in relaxed] == [
        {"T_sink": [0.0, 0.0], "h": [1.0, 1.0]},
        {"T_sink": [3.0, 3.0]},
    ]


def test_nearest_neighbor_mapping_maps_each_target_point_to_its_nearest_source() -> None:
    source = _points([[0.0], [1.0], [2.0]])
    target = _points([[1.9], [0.1]])
    mapper = NearestNeighborMapping().bind(source, target)

    received = mapper({"T": np.array([10.0, 20.0, 30.0])})

    assert received["T"].tolist() == [30.0, 10.0]


def test_nearest_neighbor_mapping_finds_the_neighbors_once_at_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searched: list[int] = []

    def counted(source: Any, target: Any) -> Any:
        searched.append(len(target))
        return nearest_neighbor(source, target)

    monkeypatch.setattr(kernels, "nearest_neighbor", counted)
    mapper = NearestNeighborMapping().bind(_points([[0.0], [1.0]]), _points([[0.9], [0.2], [1.1]]))

    passes = [mapper({"T": np.array([1.0, 2.0]) * k})["T"].tolist() for k in (1.0, 2.0, 3.0)]

    assert searched == [3]
    assert passes == [[2.0, 1.0, 2.0], [4.0, 2.0, 4.0], [6.0, 3.0, 6.0]]


def _pairwise_nearest(source: Any, target: Any) -> Any:
    """The nearest neighbours as before the blocks: one distance tensor of all pairs."""
    distances = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    return np.argmin(distances, axis=1)


@pytest.mark.parametrize("seed", range(4))
def test_nearest_neighbor_in_blocks_equals_all_pairs_at_once(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # blocks of a few target rows, so that the blocks' seams are tested
    monkeypatch.setattr(kernels, "_BLOCK_ELEMENTS", 3 * 50 * 7)
    rng = np.random.default_rng(seed)
    source = rng.random((50, 3))
    target = rng.random((333, 3))

    np.testing.assert_array_equal(
        nearest_neighbor(source, target), _pairwise_nearest(source, target)
    )


def test_nearest_neighbor_takes_the_lowest_index_on_a_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "_BLOCK_ELEMENTS", 2 * 9)
    # points on a grid: every target point is as far from two or four source points
    source = np.array([[x, y] for x in range(3) for y in range(3)], dtype=float)
    target = np.array([[x + 0.5, y + 0.5] for x in range(2) for y in range(2)] + [[0.5, 0.0]])

    indices = nearest_neighbor(source, target)

    assert indices.tolist() == [0, 1, 3, 4, 0]
    np.testing.assert_array_equal(indices, _pairwise_nearest(source, target))


def test_nearest_neighbor_never_holds_the_distances_of_all_pairs() -> None:
    rng = np.random.default_rng(0)
    source = rng.random((20_000, 3))
    target = rng.random((20_000, 3))
    tracemalloc.start()
    try:
        indices = nearest_neighbor(source, target)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert indices.shape == (20_000,)
    # all pairs at once: 20k * 20k * 3 doubles, 9.6 GB
    assert peak < 64 * 2**20, f"peak {peak / 2**20:.0f} MiB"


def test_nearest_neighbor_mapping_needs_the_points_of_both_sides() -> None:
    with pytest.raises(ValueError, match="'nearest-neighbor' needs the coordinates of both sides"):
        NearestNeighborMapping().bind(_points([[0.0]]), _points(None))


def test_a_mapping_of_a_type_is_a_new_instance_of_it() -> None:
    assert type(strategy(Mapping, "identity")) is IdentityMapping
    assert strategy(Mapping, {"type": "nearest-neighbor"}).type == "nearest-neighbor"
    assert strategy(Mapping, "nearest-neighbor") is not strategy(Mapping, "nearest-neighbor")
    with pytest.raises(
        ValueError,
        match=r"unknown mapping type 'rbf'; registered mapping types: "
        r"\['identity', 'nearest-neighbor'\]",
    ):
        strategy(Mapping, "rbf")


def test_identity_mapping_hands_the_data_over_unchanged() -> None:
    data = {"T": np.array([1.0, 2.0])}

    assert IdentityMapping().bind(_points(None), _points(None))(data) is data
    assert IdentityMapping().bind(_points([[0.0], [1.0]]), _points([[0.0], [1.0]]))(data) is data


def test_identity_mapping_needs_as_many_points_on_both_sides() -> None:
    with pytest.raises(
        ValueError, match="'identity' needs as many points on both sides: 2 sent, 3"
    ):
        IdentityMapping().bind(_points([[0.0], [1.0]]), _points([[0.0], [1.0], [2.0]]))


@pytest.mark.parametrize(
    ("current", "converged"),
    [(100.0 + 1e-4, True), (100.0 + 1e-2, False)],
    ids=["change 1e-6 relative", "change 1e-4 relative"],
)
def test_relative_convergence_compares_the_change_with_the_limit(
    current: float, converged: bool
) -> None:
    change = SideChange("T", {"T_sink": np.array([100.0])}, {"T_sink": np.array([current])})

    assert RelativeConvergence(data="T", limit=1e-5)([change]) is converged


def test_relative_convergence_ignores_other_fields() -> None:
    changes: list[Any] = [SideChange("U", {"u": np.array([1.0])}, {"u": np.array([2.0])})]

    assert RelativeConvergence(data="T", limit=1e-5)(changes) is True
