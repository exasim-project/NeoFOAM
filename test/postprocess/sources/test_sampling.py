# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the sampled-surface sources and the ``sample`` node — the numpy side.

A sampled surface reaches the value layer as three things only: face centres,
face areas, and an interpolation callable, so every behaviour below can be
pinned with a fake surface and no mesh at all. What OpenFOAM actually cuts (a
plane's area, an iso-surface's shape, the interpolated values) is a separate
question, pinned on a real case in ``test_sampling_e2e.py``.

``Sample`` reaching its surface *through the geometry* is the invariant that
keeps a declared pipeline re-evaluable every time step: the node holds a field
name, never a live mesh, which is what the fake geometries here stand in for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neofoam.postprocess import (
    Area,
    IsoSurface,
    PlaneSurface,
    Sample,
    SurfaceGeometry,
    iso_surface,
    plane,
)
from neofoam.postprocess.config import PostProcessConfig, resolve_table
from neofoam.postprocess.node import InternalDataSet, SurfaceDataSet

CASES = Path(__file__).parents[1] / "cases"

#: Two faces of a fake cut: centres a unit apart, areas that differ so a test
#: that mixed measure up with a count would fail.
FACE_CENTRES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
FACE_AREAS = np.array([0.25, 0.75])


class _FakeSurface:
    """A ``sampledSurface`` reduced to what a SurfaceGeometry reads."""

    def Cf(self) -> "np.ndarray[Any, Any]":
        return FACE_CENTRES

    def Sf(self) -> "np.ndarray[Any, Any]":
        return np.column_stack([FACE_AREAS, np.zeros(2), np.zeros(2)])

    def magSf(self) -> "np.ndarray[Any, Any]":
        return FACE_AREAS

    def area(self) -> float:
        return float(FACE_AREAS.sum())


class _FakeSampling:
    """A geometry that samples without a mesh: it hands back the name it was asked for."""

    def positions(self) -> "np.ndarray[Any, Any]":
        return FACE_CENTRES

    def face_areas(self) -> "np.ndarray[Any, Any]":
        return np.column_stack([FACE_AREAS, np.zeros(2), np.zeros(2)])

    def face_area_magnitudes(self) -> "np.ndarray[Any, Any]":
        return FACE_AREAS

    def total_area(self) -> float:
        return float(FACE_AREAS.sum())

    def sample(self, name: str) -> "np.ndarray[Any, Any]":
        return np.array([len(name), -len(name)], dtype=float)


class _FakePatchGeometry(_FakeSampling):
    """A patch: since I1 it is a sampled surface too, so it samples by name."""


class _CellGeometry:
    """A geometry with no surface behind it — cells, as an internal source builds."""

    def positions(self) -> "np.ndarray[Any, Any]":
        return FACE_CENTRES

    def volumes(self) -> "np.ndarray[Any, Any]":
        return FACE_AREAS


def _geometry(fields: dict[str, Any]) -> SurfaceGeometry:
    """A SurfaceGeometry over the fake surface, holding *fields* as the registry."""
    return SurfaceGeometry(_FakeSurface(), fields, "cellPoint")


def _dataset(geometry: Any) -> SurfaceDataSet:
    """A dataset on *geometry* whose values are the face areas — what a bare source gives."""
    return SurfaceDataSet(name="area", field=FACE_AREAS, geometry=geometry)


def _spec(rel_path: str, index: int) -> Any:
    """The table a checked-in spec file declares, loaded the way a case loads it."""
    return PostProcessConfig.load(case_dir=CASES / rel_path).tables[index]


def test_face_centres_are_the_positions_a_node_selects_on() -> None:
    assert _geometry({}).positions() == pytest.approx(FACE_CENTRES)


def test_face_areas_are_the_measure_a_node_weighs_with() -> None:
    assert _geometry({}).face_area_magnitudes() == pytest.approx(FACE_AREAS)


def test_area_of_a_sampled_surface_is_its_face_areas() -> None:
    dataset = Area().compute(_dataset(_geometry({})))

    assert dataset.field == pytest.approx(FACE_AREAS)


def test_sampling_an_unregistered_field_names_the_registered_ones() -> None:
    geometry = _geometry({"p": object()})

    with pytest.raises(KeyError, match=r"'T'.*\['p'\]"):
        geometry.sample("T")


def test_sampling_a_field_type_without_a_sampler_names_the_ones_with_one() -> None:
    # The dispatch is an isinstance check against the two samplable field
    # classes, so any other object takes the branch; the name is what it reports.
    tensor_field = type("volTensorField", (), {})()

    with pytest.raises(TypeError, match="volTensorField is not one of"):
        _geometry({"sigma": tensor_field}).sample("sigma")


def test_sample_replaces_the_values_with_what_the_geometry_interpolates() -> None:
    dataset = Sample(field="U").compute(_dataset(_FakeSampling()))

    assert dataset.field == pytest.approx([1.0, -1.0])


def test_sample_renames_the_dataset_after_the_sampled_field() -> None:
    dataset = Sample(field="U").compute(_dataset(_FakeSampling()))

    assert dataset.name == "U"


def test_sample_runs_on_a_patch_geometry() -> None:
    # A patch is a sampled surface with a frozen config, so one name-based
    # protocol serves both and ``patch(...) | Sample(...)`` resolves.
    dataset = Sample(field="U").compute(_dataset(_FakePatchGeometry()))

    assert dataset.field == pytest.approx([1.0, -1.0])


def test_sample_on_a_geometry_with_no_surface_behind_it_is_refused() -> None:
    cells = InternalDataSet(name="p", field=FACE_AREAS, geometry=_CellGeometry())

    with pytest.raises(TypeError, match="sample needs a sampled surface"):
        Sample(field="U").compute(cells)


def test_plane_sugar_builds_the_declared_plane_source() -> None:
    pipeline = plane((0.05, 0.0, 0.0), (1.0, 0.0, 0.0), field="U", scheme="cell")

    assert pipeline.source == PlaneSurface(
        point=(0.05, 0.0, 0.0), normal=(1.0, 0.0, 0.0), field="U", scheme="cell"
    )


def test_iso_surface_sugar_builds_the_declared_iso_source() -> None:
    pipeline = iso_surface("alpha.water", 0.5)

    assert pipeline.source == IsoSurface(iso_field="alpha.water", iso_value=0.5)


def test_a_bare_surface_source_samples_nothing() -> None:
    assert plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).source.field is None


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, PlaneSurface(point=(0.05, 0.0, 0.0), normal=(1.0, 0.0, 0.0))),
        (1, IsoSurface(iso_field="p", iso_value=0.05, field="U", scheme="cell")),
    ],
    ids=["plane", "isoSurface"],
)
def test_a_declared_surface_table_resolves_to_its_source(index: int, expected: Any) -> None:
    table = resolve_table(_spec("surface_tables/system/postProcess.yaml", index))

    assert table.pipeline.source == expected


def test_a_declared_sample_step_resolves_to_the_sample_node() -> None:
    table = resolve_table(_spec("surface_tables/system/postProcess.yaml", 0))

    assert table.pipeline.steps[0] == Sample(field="U")
