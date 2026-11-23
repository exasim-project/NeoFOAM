# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A transient VTKHDF file of one static unstructured mesh: :class:`VtkHdfWriter`.

The file is VTKHDF 2.1 ``UnstructuredGrid`` with ``Steps`` (ParaView >= 5.12): the
mesh once, one part per MPI rank, then the point and cell data of every time step
appended. The data of step ``i`` of a field starts at
``Steps/PointDataOffsets/<name>[i]`` (``CellDataOffsets``) in ``PointData/<name>``,
the parts' data one after the other in rank order; the mesh offsets of every step
are 0. A backend hands its mesh over as a :class:`MeshExport`.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np


class MeshExport(Protocol):
    """A mesh as VTK's unstructured-grid arrays; a backend implements it for its mesh.

    Cell ``c`` is the point indices ``connectivity[offsets[c]:offsets[c + 1]]``, of
    the VTK cell type ``cell_types[c]`` (e.g. 10 tetra, 12 hexahedron); cells of
    different types mix. Under MPI, the arrays of a rank are its piece, the point
    indices local to it. :class:`UnstructuredMesh` is one from plain arrays.
    """

    def points(self) -> np.ndarray:
        """``(n_points, 3)`` float."""
        ...

    def connectivity(self) -> np.ndarray:
        """Flat int64, the point indices of all cells."""
        ...

    def offsets(self) -> np.ndarray:
        """``(n_cells + 1,)`` int64 into :meth:`connectivity`, starting at 0."""
        ...

    def cell_types(self) -> np.ndarray:
        """``(n_cells,)`` uint8, the VTK cell type ids."""
        ...


class UnstructuredMesh:
    """A :class:`MeshExport` from arrays; it holds them, not copies. Example::

    hex_and_tet = UnstructuredMesh(
        points, connectivity=np.r_[0:8, 4, 5, 6, 8], offsets=np.array([0, 8, 12]),
        cell_types=np.array([12, 10]),
    )
    """

    def __init__(
        self,
        points: np.ndarray,
        connectivity: np.ndarray,
        offsets: np.ndarray,
        cell_types: np.ndarray,
    ) -> None:
        self._points = points
        self._connectivity = connectivity
        self._offsets = offsets
        self._cell_types = cell_types

    def points(self) -> np.ndarray:
        return self._points

    def connectivity(self) -> np.ndarray:
        return self._connectivity

    def offsets(self) -> np.ndarray:
        return self._offsets

    def cell_types(self) -> np.ndarray:
        return self._cell_types


def _mesh_arrays(mesh: MeshExport) -> dict[str, np.ndarray]:
    """The arrays of ``mesh`` in the file's dtypes; ``ValueError`` for inconsistent offsets."""
    arrays = {
        "points": np.asarray(mesh.points(), dtype=float, order="C").reshape(-1, 3),
        "connectivity": np.asarray(mesh.connectivity(), dtype=np.int64, order="C"),
        "offsets": np.asarray(mesh.offsets(), dtype=np.int64, order="C"),
        "cell_types": np.asarray(mesh.cell_types(), dtype=np.uint8, order="C"),
    }
    n_cells = len(arrays["cell_types"])
    if len(arrays["offsets"]) != n_cells + 1:
        raise ValueError(f"offsets: {len(arrays['offsets'])} entries, {n_cells + 1} expected")
    return arrays


def _prepared_mesh(path: Path, mesh: MeshExport, comm: Any) -> dict[str, np.ndarray]:
    """The mesh arrays; a failure on one rank raises on all, none is left in a collective."""
    if comm is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return _mesh_arrays(mesh)
    arrays: dict[str, np.ndarray] = {}
    error = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = _mesh_arrays(mesh)
    except Exception as caught:  # re-raised below, on every rank
        error = f"{type(caught).__name__}: {caught}"
    failed = [f"rank {r}: {e}" for r, e in enumerate(comm.allgather(error)) if e]
    if failed:
        raise ValueError(f"VtkHdfWriter mesh: {'; '.join(failed)}")
    return arrays


@contextmanager
def _no_cyclic_gc(comm: Any) -> Iterator[None]:
    """Collect on every rank at one point, then keep the collector off during the collectives."""
    if comm is None:
        yield
        return
    # Objects whose finalizers are collective (petsc4py/dolfinx objects, mpi4py communicators)
    # wait in reference cycles for the cyclic collector, which runs whenever an allocation trips
    # it: then on one rank only, possibly inside a collective below, while the other ranks wait
    # there for it, a hang. write() allocates too, so it needs this as much as the constructor.
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _native_writer() -> Any:
    """The compiled writer class; ``RuntimeError`` if this build has none."""
    try:
        return import_module("neofoam._vtkhdf").VtkHdfWriter
    except ImportError as error:
        raise RuntimeError(
            "VtkHdfWriter needs the compiled module neofoam._vtkhdf: "
            "build NeoFOAM with NEOFOAM_WITH_HDF5=ON"
        ) from error


class VtkHdfWriter:
    """Write one static mesh and its point and cell data over time to ``path`` (overwritten).

    Use it for a backend's results in ParaView without its own file format; the
    mesh comes as a :class:`MeshExport`. Every :meth:`write` opens, appends to and
    closes the file, so a run that ends without a finalizer leaves a complete file.
    With ``comm`` (an mpi4py communicator) every rank passes its piece, the
    constructor and :meth:`write` are collective and every rank writes its own part
    into the one file, in rank order; ``comm=None`` is serial. Example::

        out = VtkHdfWriter(path, UnstructuredMesh(points, connectivity, offsets, cell_types))
        out.write(0.1, point_data={"T": T}, cell_data={"p": p})
    """

    def __init__(self, path: Path, mesh: MeshExport, comm: Any = None) -> None:
        native_writer = _native_writer()
        self.path = Path(path)
        self._comm = comm  # the native writer holds its handle, this keeps it valid
        with _no_cyclic_gc(comm):
            arrays = _prepared_mesh(self.path, mesh, comm)
            handle = None if comm is None else comm.py2f()
            self._writer = native_writer(str(self.path), arrays, handle)

    def write(
        self,
        time: float,
        point_data: Optional[Mapping[str, np.ndarray]] = None,
        cell_data: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        """Append the time step ``time``; every rank passes the same field names."""
        with _no_cyclic_gc(self._comm):
            self._writer.write(time, point_data or {}, cell_data or {})
