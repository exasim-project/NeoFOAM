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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

_MESH_KEYS = ("points", "connectivity", "offsets", "cell_types")


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
    """The arrays of ``mesh`` in the file's dtypes."""
    return {
        "points": np.asarray(mesh.points(), dtype=float).reshape(-1, 3),
        "connectivity": np.asarray(mesh.connectivity(), dtype=np.int64),
        "offsets": np.asarray(mesh.offsets(), dtype=np.int64),
        "cell_types": np.asarray(mesh.cell_types(), dtype=np.uint8),
    }


def _h5py() -> Any:
    """h5py, imported on first use: an optional dependency."""
    import h5py  # noqa: PLC0415 - optional, only a run that writes VTKHDF needs it

    return h5py


def _appendable(group: Any, name: str, data: np.ndarray) -> None:
    """Append ``data`` (rows) to the resizable dataset ``group[name]``, created on first use."""
    if name not in group:
        group.create_dataset(name, data=data, maxshape=(None, *data.shape[1:]), chunks=True)
        return
    dataset = group[name]
    start = dataset.shape[0]
    dataset.resize(start + data.shape[0], axis=0)
    dataset[start:] = data


class VtkHdfWriter:
    """Write one static mesh and its point and cell data over time to ``path`` (overwritten).

    Use it for a backend's results in ParaView without its own file format; the
    mesh comes as a :class:`MeshExport`. Every :meth:`write` opens, appends to and
    closes the file, so a run that ends without a finalizer leaves a complete file.
    With ``comm`` (an mpi4py communicator) every rank passes its piece, the
    constructor and :meth:`write` are collective and rank 0 gathers and writes, one
    part per rank; ``comm=None`` is serial. Needs h5py. Example::

        out = VtkHdfWriter(path, UnstructuredMesh(points, connectivity, offsets, cell_types))
        out.write(0.1, point_data={"T": T}, cell_data={"p": p})
    """

    def __init__(self, path: Path, mesh: MeshExport, comm: Any = None) -> None:
        self.path = Path(path)
        self._comm = comm
        if comm is not None:
            # Collect on every rank before the first collective. Objects whose finalizers
            # are collective (petsc4py/dolfinx objects, mpi4py communicators) wait in
            # reference cycles for the cyclic collector, which runs whenever an allocation
            # trips it: then on one rank only, possibly inside the gather below, while the
            # other ranks wait there for it, a hang. Here every rank collects at one point.
            gc.collect()
        parts = self._gather(_mesh_arrays(mesh))
        self._sizes = {
            "Point": [len(part["points"]) for part in parts],
            "Cell": [len(part["cell_types"]) for part in parts],
        }
        if parts:
            self._create(parts)

    def write(
        self,
        time: float,
        point_data: Optional[Mapping[str, np.ndarray]] = None,
        cell_data: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        """Append the time step ``time``; every rank passes the same field names."""
        data = {"Point": dict(point_data or {}), "Cell": dict(cell_data or {})}
        parts = self._gather(data)
        if parts:
            self._append(time, parts)

    def _gather(self, piece: dict[str, Any]) -> list[dict[str, Any]]:
        """Every rank's ``piece`` on rank 0 in rank order, ``[]`` on the others."""
        if self._comm is None:
            return [piece]
        return self._comm.gather(piece, root=0) or []

    def _create(self, parts: list[dict[str, np.ndarray]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _h5py().File(self.path, "w") as f:
            root = f.create_group("VTKHDF")
            root.attrs["Version"] = (2, 1)
            kind = "UnstructuredGrid"
            root.attrs.create("Type", np.bytes_(kind), dtype=f"S{len(kind)}")
            root.create_dataset("NumberOfPoints", data=self._sizes["Point"], dtype=np.int64)
            root.create_dataset("NumberOfCells", data=self._sizes["Cell"], dtype=np.int64)
            connectivity_ids = [len(part["connectivity"]) for part in parts]
            root.create_dataset("NumberOfConnectivityIds", data=connectivity_ids, dtype=np.int64)
            for name, key in zip(("Points", "Connectivity", "Offsets", "Types"), _MESH_KEYS):
                root.create_dataset(name, data=np.concatenate([part[key] for part in parts]))
            root.create_group("PointData")
            root.create_group("CellData")
            steps = root.create_group("Steps")
            steps.attrs["NSteps"] = 0
            steps.create_group("PointDataOffsets")
            steps.create_group("CellDataOffsets")

    def _append(self, time: float, parts: list[dict[str, dict[str, np.ndarray]]]) -> None:
        with _h5py().File(self.path, "a") as f:
            root = f["VTKHDF"]
            steps = root["Steps"]
            _appendable(steps, "Values", np.array([time], dtype=float))
            _appendable(steps, "NumberOfParts", np.array([len(parts)], dtype=np.int64))
            # one static mesh: every step reads the same parts, points and cells
            for name in ("PartOffsets", "PointOffsets", "CellOffsets"):
                _appendable(steps, name, np.zeros(1, dtype=np.int64))
            _appendable(steps, "ConnectivityIdOffsets", np.zeros((1, 1), dtype=np.int64))
            for kind in ("Point", "Cell"):
                for name in parts[0][kind]:
                    values = self._concatenate(parts, kind, name)
                    start = root[f"{kind}Data"][name].shape[0] if name in root[f"{kind}Data"] else 0
                    _appendable(steps[f"{kind}DataOffsets"], name, np.array([start], np.int64))
                    _appendable(root[f"{kind}Data"], name, values)
            steps.attrs["NSteps"] = int(steps["Values"].shape[0])

    def _concatenate(self, parts: list[dict[str, Any]], kind: str, name: str) -> np.ndarray:
        """The parts' values of the field ``name``; ``ValueError`` for a wrong row count."""
        arrays = [np.asarray(part[kind][name], dtype=float) for part in parts]
        rows = [len(array) for array in arrays]
        if rows != self._sizes[kind]:
            raise ValueError(
                f"{kind.lower()} data {name}: {rows} rows per part, {self._sizes[kind]} expected"
            )
        return np.concatenate(arrays)
