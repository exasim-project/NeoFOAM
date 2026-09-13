# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Geometry adapters: the cells, a sampled surface, one boundary patch, a set of points.

Every adapter satisfies the :class:`~neofoam.postprocess.node.Geometry` protocol
(``positions`` and an optional ``measure``), and all but :class:`CellGeometry`
also satisfy :class:`~neofoam.postprocess.node.SamplingGeometry` — they can
interpolate a *registered field name* onto their elements, so a node never
learns which one it is working on. :class:`CellGeometry` needs nothing but an
fvMesh-shaped object; the others build a pybFoam ``sampledSurface`` or
``sampledSet``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import pybFoam as pyf
from pybFoam import sampling

#: The interpolation a patch is sampled with: it hands back the patch's own
#: boundary values instead of interpolating them from the adjacent cells.
_PATCH_INTERPOLATION = "cell"

#: The interpolation sample points are sampled with: a point that lands on a wall
#: then carries the boundary condition and not the adjacent cell's value.
_POINT_INTERPOLATION = "cellPoint"

#: pybFoam volume-field class -> the suffix of its sampling functions.
_SAMPLER_KINDS = {pyf.volScalarField: "Scalar", pyf.volVectorField: "Vector"}


def _patch_names(mesh: Any) -> list[str]:
    """The mesh's boundary patch names, by index (iterating fvBoundaryMesh crashes nanobind)."""
    boundary = mesh.boundary()
    return [str(boundary[index].name()) for index in range(len(boundary))]


def _sample(
    target: Any, sampler: str, fields: Mapping[str, Any], scheme: str, name: str
) -> np.ndarray[Any, Any]:
    """Interpolate the registered field *name* onto *target* with ``<sampler><kind>``."""
    if name not in fields:
        raise KeyError(
            f"postProcess: field {name!r} is not registered; available: {sorted(fields)}"
        )
    field = fields[name]
    kind = next((suffix for cls, suffix in _SAMPLER_KINDS.items() if isinstance(field, cls)), None)
    if kind is None:
        raise TypeError(
            f"postProcess: cannot sample {name!r}; {type(field).__name__} is not one of "
            f"{sorted(cls.__name__ for cls in _SAMPLER_KINDS)}"
        )
    interpolator = getattr(sampling, f"interpolation{kind}").New(pyf.Word(scheme), field)
    return np.asarray(getattr(sampling, f"{sampler}{kind}")(target, interpolator))


class CellGeometry:
    """Cell centres and volumes, duck-typed over a pybFoam-like fvMesh.

    The geometry of :class:`~neofoam.postprocess.sources.fields.InternalField`, and the one
    adapter that touches no pybFoam symbol: it calls ``C()`` and ``V()`` on
    whatever mesh it is handed, so a fake mesh is enough to exercise it. It has
    no ``sample``, which is what makes
    ``field("p") | Sample(field="U")`` a named error instead of a crash::

        CellGeometry(ctx.mesh).measure.sum()  # the mesh volume
    """

    def __init__(self, mesh: Any) -> None:
        self._mesh = mesh

    @property
    def positions(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._mesh.C().internalField())

    @property
    def measure(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._mesh.V())


class SurfaceGeometry:
    """Face centres and areas of a sampled surface, plus the interpolation onto it.

    What a surface source hands its nodes: ``positions`` are the face centres,
    ``measure`` the face areas (so ``| Area() | Sum()`` is the cut area), and
    :meth:`sample` interpolates a registered field onto those faces. Holding the
    field registry here is what keeps
    :class:`~neofoam.postprocess.sources.sampling.Sample` free of a live mesh — a
    pipeline is declared once and re-resolved every time step::

        SurfaceGeometry(surface, ctx.fields, "cellPoint").sample("U")
    """

    def __init__(self, surface: Any, fields: Mapping[str, Any], scheme: str) -> None:
        self._surface = surface
        self._fields = fields
        self._scheme = scheme

    @property
    def positions(self) -> np.ndarray[Any, Any]:
        """Face centres, shape ``(n, 3)``."""
        return np.asarray(self._surface.Cf())

    @property
    def measure(self) -> np.ndarray[Any, Any]:
        """Face areas, shape ``(n,)``."""
        return np.asarray(self._surface.magSf())

    def sample(self, name: str) -> np.ndarray[Any, Any]:
        """Interpolate the registered field *name* onto the face centres."""
        return _sample(self._surface, "sampleOnFaces", self._fields, self._scheme, name)


class PatchGeometry(SurfaceGeometry):
    """One boundary patch of an fvMesh, as a sampled surface.

    The geometry of :class:`~neofoam.postprocess.sources.fields.PatchField` — use it for
    a table over a wall or an inlet, where the measure is the face area and
    :class:`~neofoam.postprocess.nodes.aggregators.SurfIntegrate` is the aggregator.
    pybFoam binds no ``boundaryField()`` accessor, so a patch is a
    ``sampledSurface`` of type ``patch`` with the config frozen
    (``triangulate=False``, ``cell`` interpolation, so a row is a face of the
    mesh and the sampled values are the patch's own boundary values). The
    surface is built once, at construction, and an unknown patch name — or one
    that samples no face at all — raises there::

        PatchGeometry(ctx.mesh, "movingWall", ctx.fields).measure.sum()  # the patch area
    """

    def __init__(self, mesh: Any, patch_name: str, fields: Mapping[str, Any]) -> None:
        available = _patch_names(mesh)
        if patch_name not in available:
            raise KeyError(
                f"postProcess: patch {patch_name!r} is not on the mesh; available: {available}"
            )
        spec = sampling.SampledPatchConfig(patches=[patch_name], triangulate=False)
        surface = sampling.sampledSurface.New(pyf.Word(patch_name), mesh, spec.to_foam_dict())
        surface.update()
        # sampledPatch drops a patch it cannot sample (an ``empty`` one) with a
        # debug message only, and every row of the table would then read zero.
        if np.asarray(surface.magSf()).size == 0:
            raise ValueError(
                f"postProcess: patch {patch_name!r} samples no faces — an 'empty' patch "
                f"carries no data to post-process"
            )
        super().__init__(surface, fields, _PATCH_INTERPOLATION)


class PointGeometry:
    """The sample points of a pybFoam ``sampledSet`` — positions, no measure.

    The geometry of :class:`~neofoam.postprocess.sources.fields.LineField`: a probe has
    no area or volume, so ``measure`` is ``None`` and the measure-weighted
    aggregators reject it — reduce it with
    :class:`~neofoam.postprocess.nodes.aggregators.Mean` or write every point with
    :class:`~neofoam.postprocess.nodes.rows.Rows`. OpenFOAM drops the points of the
    spec that fall outside the mesh, so ``positions`` may be shorter than the
    spec asked for::

        PointGeometry(ctx.mesh, spec.to_foam_dict(), "profile", ctx.fields).positions
    """

    def __init__(self, mesh: Any, set_dict: Any, name: str, fields: Mapping[str, Any]) -> None:
        # The sampledSet keeps a reference to the search engine, so it has to
        # outlive this constructor.
        self._search = sampling.meshSearch(mesh)
        self._set = sampling.sampledSet.New(pyf.Word(name), mesh, self._search, set_dict)
        self._fields = fields

    @property
    def positions(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._set.points())

    @property
    def measure(self) -> Optional[np.ndarray[Any, Any]]:
        return None

    def sample(self, name: str) -> np.ndarray[Any, Any]:
        """Interpolate the registered field *name* at the sample points."""
        return _sample(self._set, "sampleSet", self._fields, _POINT_INTERPOLATION, name)
