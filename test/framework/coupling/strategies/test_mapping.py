# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The mappings of a coupling, each on a few hand-checked values."""

import tracemalloc
from typing import Any, cast

import numpy as np
import pytest

from neofoam.framework.coupling import (
    CouplingMapping,
    IdentityMapping,
    NearestNeighborMapping,
)
from neofoam.framework.coupling.location import Geometry
from neofoam.framework.coupling.strategies import mapping as mapping_module
from neofoam.framework.coupling.strategies.mapping import nearest_neighbor


def _points(coordinates: Any) -> Geometry:
    """The geometry of a port with ``coordinates`` (``None``: without points)."""
    return Geometry() if coordinates is None else Geometry.of(coordinates, None)


def test_nearest_neighbor_picks_the_closest_source_point() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    target = np.array([[1.9, 0.1], [0.2, 0.0], [1.1, -0.3]])

    assert nearest_neighbor(source, target).tolist() == [2, 0, 1]


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

    monkeypatch.setattr(mapping_module, "nearest_neighbor", counted)
    mapper = NearestNeighborMapping().bind(_points([[0.0], [1.0]]), _points([[0.9], [0.2], [1.1]]))

    iterations = [mapper({"T": np.array([1.0, 2.0]) * k})["T"].tolist() for k in (1.0, 2.0, 3.0)]

    assert searched == [3]
    assert iterations == [[2.0, 1.0, 2.0], [4.0, 2.0, 4.0], [6.0, 3.0, 6.0]]


def _pairwise_nearest(source: Any, target: Any) -> Any:
    """The nearest neighbours from one distance tensor of all pairs."""
    distances = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    return np.argmin(distances, axis=1)


@pytest.mark.parametrize("seed", range(4))
def test_nearest_neighbor_in_blocks_equals_all_pairs_at_once(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # blocks of a few target rows, so that the blocks' seams are tested
    monkeypatch.setattr(mapping_module, "_BLOCK_ELEMENTS", 3 * 50 * 7)
    rng = np.random.default_rng(seed)
    source = rng.random((50, 3))
    target = rng.random((333, 3))

    np.testing.assert_array_equal(
        nearest_neighbor(source, target), _pairwise_nearest(source, target)
    )


def test_nearest_neighbor_takes_the_lowest_index_on_a_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapping_module, "_BLOCK_ELEMENTS", 2 * 9)
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


def test_nearest_neighbor_mapping_needs_the_points_of_both_ports() -> None:
    with pytest.raises(ValueError, match="the receiving port has none: give its mesh geometry"):
        NearestNeighborMapping().bind(_points([[0.0]]), _points(None))
    with pytest.raises(ValueError, match="the sending port has none"):
        NearestNeighborMapping().bind(_points(None), _points([[0.0]]))
    with pytest.raises(ValueError, match="neither port has any.*or choose mapping 'identity'"):
        NearestNeighborMapping().bind(_points(None), _points(None))


@pytest.mark.parametrize(
    ("options", "mapping"),
    [
        ({"type": "identity"}, IdentityMapping),
        ({"type": "nearest-neighbor"}, NearestNeighborMapping),
    ],
)
def test_a_mapping_is_selected_by_its_type(options: dict[str, str], mapping: type) -> None:
    selected = cast(Any, CouplingMapping).create(mapping=options).mapping

    assert type(selected) is mapping


def test_identity_mapping_hands_the_data_over_unchanged() -> None:
    data = {"T": np.array([1.0, 2.0])}

    assert IdentityMapping().bind(_points(None), _points(None))(data) is data
    assert IdentityMapping().bind(_points([[0.0], [1.0]]), _points([[0.0], [1.0]]))(data) is data


def test_identity_mapping_needs_as_many_entities_on_both_ports() -> None:
    with pytest.raises(
        ValueError, match="'identity' needs as many entities on both ports: 2 sent, 3"
    ):
        IdentityMapping().bind(_points([[0.0], [1.0]]), _points([[0.0], [1.0], [2.0]]))


# a patch of three faces at x = 0, 1, 2 with the areas 1, 2, 3, and two cells
FACES = Geometry.of([[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0])


CELLS = Geometry.of([[0.0], [1.0]], [1.0, 3.0])


VALUE = Geometry(size=1)


@pytest.mark.parametrize(
    ("options", "source", "sent", "received"),
    [
        ({"type": "area-mean"}, FACES, [1.0, 2.0, 3.0], 14.0 / 6.0),
        ({"type": "area-integral"}, FACES, [1.0, 2.0, 3.0], 14.0),
        ({"type": "volume-mean"}, CELLS, [2.0, 6.0], 20.0 / 4.0),
        ({"type": "volume-integral"}, CELLS, [2.0, 6.0], 20.0),
    ],
    ids=["area-mean", "area-integral", "volume-mean", "volume-integral"],
)
def test_a_reduction_weighs_each_value_by_its_measure(
    options: dict[str, str], source: Geometry, sent: list[float], received: float
) -> None:
    mapping = cast(Any, CouplingMapping).create(mapping=options).mapping

    reduced = mapping.bind(source, VALUE)({"q": np.array(sent)})

    np.testing.assert_allclose(reduced["q"], received, rtol=1e-14, err_msg=options["type"])


@pytest.mark.parametrize("options", [{"type": "area-mean"}, {"type": "uniform"}])
def test_a_value_mapping_takes_scalar_fields_only(options: dict[str, str]) -> None:
    mapping = cast(Any, CouplingMapping).create(mapping=options).mapping
    vector = {"U": np.ones((3, 3))} if options["type"] != "uniform" else {"U": np.ones(3)}

    with pytest.raises(ValueError, match=f"mapping '{options['type']}' takes scalar"):
        mapping.bind(FACES, FACES)(vector)


@pytest.mark.parametrize("target", [FACES, CELLS], ids=["faces", "cells"])
def test_uniform_puts_one_value_on_every_entity(target: Geometry) -> None:
    mapping = cast(Any, CouplingMapping).create(mapping={"type": "uniform"}).mapping

    spread = mapping.bind(VALUE, target)({"T": np.array(300.0)})

    assert spread["T"].tolist() == [300.0] * int(target.size or 0)


def test_probe_takes_the_value_of_the_cell_nearest_each_point() -> None:
    mapping = (
        cast(Any, CouplingMapping)
        .create(mapping={"type": "probe", "points": [[0.9], [0.2], [0.6]]})
        .mapping
    )

    probed = mapping.bind(CELLS, VALUE)({"T": np.array([10.0, 20.0])})

    assert probed["T"].tolist() == [20.0, 10.0, 20.0]


@pytest.mark.parametrize(
    ("mapping", "sent", "received", "error"),
    [
        (NearestNeighborMapping(), "face", "value", "maps face -> face, cell -> cell, not face"),
        (IdentityMapping(), "cell", "face", "not cell -> face"),
        ({"type": "area-mean"}, "cell", "value", "maps face -> value, not cell -> value"),
        ({"type": "uniform"}, "face", "face", "maps value -> face, value -> cell, not face"),
        ({"type": "probe", "points": [[0.0]]}, "face", "probe", "maps cell -> probe, not"),
    ],
    ids=["nearest-neighbor", "identity", "area-mean", "uniform", "probe"],
)
def test_a_mapping_checks_the_entities_it_maps(
    mapping: Any, sent: str, received: str, error: str
) -> None:
    if isinstance(mapping, dict):
        mapping = cast(Any, CouplingMapping).create(mapping=mapping).mapping

    with pytest.raises(ValueError, match=error):
        mapping.check(sent, received)
