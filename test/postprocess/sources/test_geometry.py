# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the geometry adapters — what needs no live mesh.

``CellGeometry`` reads nothing but ``C()`` and ``V()``, so a fake mesh is the
whole backend, and the patch-name check of ``PatchGeometry`` runs before the
``sampledSurface`` is built, so a fake ``boundary()`` is enough to prove the
error message names what the mesh actually carries. Its second check — a patch
that samples no face, which is how ``sampledPatch`` silently reports an
``empty`` patch — needs a surface, but only one that answers ``magSf()``. The
values a real patch or a real point set hand back are pinned in
``test_sources_e2e.py``, which has an OpenFOAM case behind it.

The one contract that cannot be faked is the *shape* of the sampling protocol:
every samplable geometry takes a registered field **name**, never a live pybFoam
field, because ``Sample`` holds a name and no mesh. Two adapters disagreeing on
that is a bug the type checker cannot see, so the signature itself is asserted.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neofoam.postprocess import sources
from neofoam.postprocess.node import SamplingGeometry
from neofoam.postprocess.sources.geometry import (
    CellGeometry,
    PatchGeometry,
    PointGeometry,
    SurfaceGeometry,
)

CELL_CENTRES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
CELL_VOLUMES = np.array([0.5, 0.25, 1.0])

PATCH_NAMES = ("movingWall", "fixedWalls", "frontAndBack")


class FakeCellField:
    """A volume field: the cell geometry reads only its internal values."""

    def __init__(self, values: "np.ndarray[Any, Any]") -> None:
        self._values = values

    def internalField(self) -> "np.ndarray[Any, Any]":
        return self._values


class FakePatch:
    """An fvPatch: the patch lookup reads only its name."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class FakeEmptySurface:
    """A sampledSurface that kept no face — what an ``empty`` patch samples as."""

    def update(self) -> None:
        pass

    def magSf(self) -> "np.ndarray[Any, Any]":
        return np.zeros(0)


class FakeMesh:
    """An fvMesh: cell centres, cell volumes and an indexable boundary mesh."""

    def C(self) -> FakeCellField:
        return FakeCellField(CELL_CENTRES)

    def V(self) -> "np.ndarray[Any, Any]":
        return CELL_VOLUMES

    def boundary(self) -> list[FakePatch]:
        return [FakePatch(name) for name in PATCH_NAMES]


def test_cell_geometry_positions_are_the_cell_centres() -> None:
    geometry = CellGeometry(FakeMesh())

    assert_allclose(geometry.positions, CELL_CENTRES, rtol=0, atol=0)


def test_cell_geometry_measure_is_the_cell_volumes() -> None:
    geometry = CellGeometry(FakeMesh())

    assert_allclose(geometry.measure, CELL_VOLUMES, rtol=0, atol=0)


def test_patch_geometry_rejects_a_patch_the_mesh_does_not_have() -> None:
    with pytest.raises(KeyError, match="movingWall"):
        PatchGeometry(FakeMesh(), "inlet", {})


def test_patch_geometry_rejects_a_patch_that_samples_no_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # sampledPatch drops an ``empty`` patch with a debug message only, so the
    # sampled surface comes back with no faces at all
    monkeypatch.setattr(
        sources.geometry.sampling,
        "sampledSurface",
        SimpleNamespace(New=lambda *args: FakeEmptySurface()),
    )

    with pytest.raises(ValueError, match="frontAndBack"):
        PatchGeometry(FakeMesh(), "frontAndBack", {})


def test_a_patch_is_a_sampled_surface() -> None:
    assert issubclass(PatchGeometry, SurfaceGeometry)


@pytest.mark.parametrize(
    "geometry",
    [SurfaceGeometry, PatchGeometry, PointGeometry],
    ids=lambda cls: cls.__name__,
)
def test_a_samplable_geometry_samples_a_registered_field_name(geometry: type) -> None:
    assert list(inspect.signature(geometry.sample).parameters) == ["self", "name"]


@pytest.mark.parametrize(
    "geometry",
    [SurfaceGeometry, PatchGeometry, PointGeometry],
    ids=lambda cls: cls.__name__,
)
def test_a_samplable_geometry_satisfies_the_sampling_protocol(geometry: type) -> None:
    assert issubclass(geometry, SamplingGeometry)


def test_the_cell_geometry_is_not_samplable() -> None:
    assert not isinstance(CellGeometry(FakeMesh()), SamplingGeometry)
