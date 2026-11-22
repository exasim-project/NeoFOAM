# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Geometry adapters: the cells, a sampled surface, one boundary patch, a set of points.

Every adapter satisfies the protocol its dataset declares —
:class:`~neofoam.postprocess.node.InternalMesh`,
:class:`~neofoam.postprocess.node.BoundaryMesh`,
:class:`~neofoam.postprocess.node.SurfaceMesh` or
:class:`~neofoam.postprocess.node.SetGeometry` — and all but the two cell adapters
also satisfy :class:`~neofoam.postprocess.node.SamplingGeometry` — they can
interpolate a *registered field name* onto their elements, so a node never
learns which one it is working on. :class:`CellGeometry` and
:class:`NeonCellGeometry` need nothing but an fvMesh-shaped object, and differ
only in where they put the numbers (host, or a NeoN executor); the others build
a pybFoam ``sampledSurface`` or ``sampledSet``, which are host-only.
"""

from __future__ import annotations

import atexit
from collections.abc import Mapping
from typing import Any

import numpy as np
import pybFoam as pyf
from pybFoam import sampling

from neofoam.postprocess._reduce import reduce_sum

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

    The geometry of :class:`~neofoam.postprocess.sources.fields.InternalField` for a
    pybFoam field, and the one adapter that touches no pybFoam symbol: it calls
    ``C()`` and ``V()`` on whatever mesh it is handed, so a fake mesh is enough
    to exercise it. It has no ``sample``, which is what makes
    ``field("p") | Sample(field="U")`` a named error instead of a crash::

        CellGeometry(ctx.mesh).volumes().sum()  # the mesh volume
    """

    def __init__(self, mesh: Any) -> None:
        self._mesh = mesh

    def positions(self) -> np.ndarray[Any, Any]:
        """Cell centres, shape ``(n, 3)``."""
        return np.asarray(self._mesh.C().internalField())

    def volumes(self) -> np.ndarray[Any, Any]:
        """Cell volumes, shape ``(n,)``."""
        return np.asarray(self._mesh.V())


class NeonCellGeometry:
    """Cell centres and volumes as NeoN vectors on one executor.

    The geometry a NeoN volume field is post-processed on: a kernel call needs
    every array on the executor its values live on, so the mesh's ``C()`` and
    ``V()`` — which are host data even on a NeoN run — are copied there once.
    Build it through :func:`neon_cell_geometry`, which keeps the copy for the
    rest of the run; construct it directly only to pin a second executor::

        neon_cell_geometry(ctx.mesh, field.internal_vector().exec()).volumes()
    """

    def __init__(self, mesh: Any, executor: Any) -> None:
        import neon  # noqa: PLC0415  # NeoN is only needed on the NeoN path

        centres = np.asarray(mesh.C().internalField())
        self._positions = neon.VectorVector(executor, [neon.Vec3(*centre) for centre in centres])
        self._volumes = neon.ScalarVector(executor, np.asarray(mesh.V(), dtype=float))

    def positions(self) -> Any:
        """Cell centres as a NeoN ``VectorVector``."""
        return self._positions

    def volumes(self) -> Any:
        """Cell volumes as a NeoN ``ScalarVector``."""
        return self._volumes


#: The cell geometry already mirrored onto an executor, by mesh and executor.
#: Post-processing runs on a static mesh, so the copy is made once per run and
#: not per write step; a mesh that moved would need this dropped. The mesh is
#: kept beside its geometry: the key is the mesh's address, which another object
#: could take over once the mesh is collected.
_NEON_CELL_GEOMETRIES: dict[tuple[int, str], tuple[Any, NeonCellGeometry]] = {}


#: Whether the cache is already set to empty itself at exit (see below).
_RELEASE_REGISTERED = False


def neon_cell_geometry(mesh: Any, executor: Any) -> NeonCellGeometry:
    """The cells of *mesh* on *executor*, built on first use and kept afterwards."""
    key = (id(mesh), repr(executor))
    cached = _NEON_CELL_GEOMETRIES.get(key)
    if cached is None:
        _release_the_cache_before_kokkos_finalizes()
        cached = (mesh, NeonCellGeometry(mesh, executor))
        _NEON_CELL_GEOMETRIES[key] = cached
    return cached[1]


def _release_the_cache_before_kokkos_finalizes() -> None:
    """Have the cache drop its vectors at exit, while Kokkos is still up.

    NeoN's own ``atexit`` handler collects the Python-owned NeoN objects and
    then calls ``Kokkos::finalize``; a vector still held here would be
    deallocated after that and Kokkos aborts the process. ``atexit`` runs its
    handlers last-registered-first, and this one is registered on the first NeoN
    field — long after NeoN was initialized — so it runs before that handler.
    """
    global _RELEASE_REGISTERED
    if _RELEASE_REGISTERED:
        return
    atexit.register(_NEON_CELL_GEOMETRIES.clear)
    _RELEASE_REGISTERED = True


class SurfaceGeometry:
    """Face centres and areas of a sampled surface, plus the interpolation onto it.

    What a surface source hands its nodes: :meth:`positions` are the face
    centres, :meth:`face_area_magnitudes` the face areas (so ``| Area() |
    Sum()`` is the cut area), and :meth:`sample` interpolates a registered field
    onto those faces. Holding the field registry here is what keeps
    :class:`~neofoam.postprocess.sources.sampling.Sample` free of a live mesh — a
    pipeline is declared once and re-resolved every time step::

        SurfaceGeometry(surface, ctx.fields, "cellPoint").sample("U")
    """

    def __init__(self, surface: Any, fields: Mapping[str, Any], scheme: str) -> None:
        self._surface = surface
        self._fields = fields
        self._scheme = scheme

    def positions(self) -> np.ndarray[Any, Any]:
        """Face centres, shape ``(n, 3)``."""
        return np.asarray(self._surface.Cf())

    def face_areas(self) -> np.ndarray[Any, Any]:
        """Face area vectors, shape ``(n, 3)``."""
        return np.asarray(self._surface.Sf())

    def face_area_magnitudes(self) -> np.ndarray[Any, Any]:
        """Face areas, shape ``(n,)``."""
        return np.asarray(self._surface.magSf())

    def total_area(self) -> float:
        """The area of the whole surface."""
        return float(self._surface.area())

    def sample(self, name: str) -> np.ndarray[Any, Any]:
        """Interpolate the registered field *name* onto the face centres."""
        return _sample(self._surface, "sampleOnFaces", self._fields, self._scheme, name)


def _global_face_count(surface: Any) -> int:
    """How many faces a sampled surface has over all ranks; its own, in serial.

    Decomposed, a valid patch legitimately has no face on some ranks — one side of
    a box is owned by a few subdomains and absent from the rest — so local
    emptiness says nothing. Only a patch that samples nothing *anywhere* is the
    unusable one. The reduction is collective, so it has to be reached by every
    rank and not only by those about to complain (R6).
    """
    local = float(np.asarray(surface.magSf()).size)
    return int(reduce_sum(np.array([local]))[0])


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
    that samples no face on any rank — raises there::

        PatchGeometry(ctx.mesh, "movingWall", ctx.fields).total_area()
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
        if _global_face_count(surface) == 0:
            raise ValueError(
                f"postProcess: patch {patch_name!r} samples no face on any rank — an "
                f"'empty' patch carries no data to post-process"
            )
        super().__init__(surface, fields, _PATCH_INTERPOLATION)


class PointGeometry:
    """The sample points of a pybFoam ``sampledSet`` — positions and distances, no measure.

    The geometry of :class:`~neofoam.postprocess.sources.fields.LineField`: a probe has
    no area or volume, so it offers neither and the measure-weighted
    aggregators reject it by name — reduce it with
    :class:`~neofoam.postprocess.nodes.aggregators.Mean` or write every point with
    :class:`~neofoam.postprocess.nodes.rows.Rows`. OpenFOAM drops the points of the
    spec that fall outside the mesh, so :meth:`positions` may be shorter than the
    spec asked for::

        PointGeometry(ctx.mesh, spec.to_foam_dict(), "profile", ctx.fields).positions()
    """

    def __init__(self, mesh: Any, set_dict: Any, name: str, fields: Mapping[str, Any]) -> None:
        # The sampledSet keeps a reference to the search engine, so it has to
        # outlive this constructor.
        self._search = sampling.meshSearch(mesh)
        self._set = sampling.sampledSet.New(pyf.Word(name), mesh, self._search, set_dict)
        self._fields = fields

    def positions(self) -> np.ndarray[Any, Any]:
        """Sample points, shape ``(n, 3)``."""
        return np.asarray(self._set.points())

    def distance(self) -> np.ndarray[Any, Any]:
        """Distance along the set, shape ``(n,)``."""
        return np.asarray(self._set.distance())

    def sample(self, name: str) -> np.ndarray[Any, Any]:
        """Interpolate the registered field *name* at the sample points."""
        return _sample(self._set, "sampleSet", self._fields, _POINT_INTERPOLATION, name)
