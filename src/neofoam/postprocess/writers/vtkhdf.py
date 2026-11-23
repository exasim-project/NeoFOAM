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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

# kind, name, rows, expected rows, shape after the rows
_FieldLayout = tuple[str, str, int, int, tuple[int, ...]]


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


def _checked_field(name: Any, values: Any) -> np.ndarray:
    """``values`` as float rows; raises for a name HDF5 cannot store or a scalar."""
    if not isinstance(name, str):
        raise TypeError(f"field name {name!r} is not a string")
    name.encode()
    if not name or name == "." or "/" in name or "\0" in name:
        raise ValueError(f"invalid field name '{name}'")
    array = np.asarray(values, dtype=np.float64, order="C")
    if array.ndim == 0:
        raise ValueError("a field needs one row per point or cell, got a scalar")
    return array


def _step_fields(point_data: Any, cell_data: Any) -> dict[str, dict[str, np.ndarray]]:
    """Each kind's fields as float arrays, sorted by name: the order of the collectives."""
    fields = {}
    for kind, data in (("point", point_data), ("cell", cell_data)):
        arrays = {name: _checked_field(name, values) for name, values in data.items()}
        fields[kind] = dict(sorted(arrays.items()))
    return fields


def _check_same_fields(reports: list[tuple[str, list[_FieldLayout]]]) -> None:
    """``ValueError`` if any rank failed or the ranks pass different field names."""
    errors = "".join(f"rank {r}: {error}; " for r, (error, _) in enumerate(reports) if error)
    if errors:
        raise ValueError(f"VtkHdfWriter.write: {errors}")
    names = [[field[:2] for field in layout] for _, layout in reports]
    if any(other != names[0] for other in names):
        listed = (
            f"\nrank {r}:\n" + "".join(f"{kind} {name}\n" for kind, name in rank_names)
            for r, rank_names in enumerate(names)
        )
        raise ValueError(f"VtkHdfWriter.write: the ranks pass different fields:{''.join(listed)}")


def _agreed_components(per_rank: Sequence[_FieldLayout]) -> tuple[int, ...]:
    """A field's shape after the rows; a rank without rows may pass any shape."""
    label = f"{per_rank[0][0]} data {per_rank[0][1]}"
    with_rows = {components for _, _, rows, _, components in per_rank if rows > 0}
    if len(with_rows) > 1:
        raise ValueError(f"{label}: the ranks' components differ")
    rows = [field[2] for field in per_rank]
    expected = [field[3] for field in per_rank]
    if rows != expected:
        raise ValueError(f"{label}: {rows} rows per part, {expected} expected")
    components = with_rows.pop() if with_rows else per_rank[0][4]
    if 0 in components:
        raise ValueError(f"{label}: a component dimension is zero")
    return components


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
            self._writer = native_writer(str(self.path), **arrays, comm=handle)
        self._rows = {"point": len(arrays["points"]), "cell": len(arrays["cell_types"])}
        self._components: dict[tuple[str, str], tuple[int, ...]] = {}

    def write(
        self,
        time: float,
        point_data: Optional[Mapping[str, np.ndarray]] = None,
        cell_data: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        """Append the time step ``time``; every rank passes the same field names."""
        with _no_cyclic_gc(self._comm):
            value, fields = self._validated(time, point_data or {}, cell_data or {})
            self._writer.write(value, fields["point"], fields["cell"])

    def _validated(
        self, time: Any, point_data: Any, cell_data: Any
    ) -> tuple[float, dict[str, list[tuple[str, np.ndarray, tuple[int, ...]]]]]:
        """The step for the native writer; the same ``ValueError`` on every rank if it is bad."""
        value = 0.0
        fields: dict[str, dict[str, np.ndarray]] = {"point": {}, "cell": {}}
        error = ""
        try:
            value = float(time)
            fields = _step_fields(point_data, cell_data)
        except Exception as caught:  # re-raised below, on every rank
            error = str(caught)
        layout = [
            (kind, name, len(array), self._rows[kind], array.shape[1:])
            for kind, arrays in fields.items()
            for name, array in arrays.items()
        ]
        reports = [(error, layout)] if self._comm is None else self._comm.allgather((error, layout))
        _check_same_fields(reports)
        components = self._agreed_step_components(reports)
        return value, {
            kind: [(name, array, components[kind, name]) for name, array in arrays.items()]
            for kind, arrays in fields.items()
        }

    def _agreed_step_components(
        self, reports: list[tuple[str, list[_FieldLayout]]]
    ) -> dict[tuple[str, str], tuple[int, ...]]:
        """Every field's components, which must not change between steps."""
        agreed = {}
        for per_rank in zip(*(layout for _, layout in reports)):
            key = per_rank[0][:2]
            agreed[key] = _agreed_components(per_rank)
            if self._components.get(key, agreed[key]) != agreed[key]:
                raise ValueError(f"{key[0]} data {key[1]}: components changed between steps")
        self._components.update(agreed)
        return agreed
