# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""VTKHDF output in several files, one per group of processes: :class:`VtkHdfFilesWriter`.

The processes form groups (one per process, one per node, or one per ``k``
consecutive ranks); the leader of a group gathers its members' parts and writes
``<stem>/<stem>_<g>.vtkhdf`` with serial h5py, the same VTKHDF 2.1
``UnstructuredGrid`` with ``Steps`` as :class:`~.vtkhdf.VtkHdfWriter`, one part per
member. Rank 0 writes the index ``path``: also such a file, of all parts, whose
arrays are HDF5 virtual datasets over the group files (sources named relative to
the index's directory, so the directory can move), so ParaView or ``vtkHDFReader``
read it as the one file :class:`~.vtkhdf.VtkHdfWriter` would write. The index is
rewritten (to a temporary file, then renamed) after every step, so all files are
valid after every :meth:`~VtkHdfFilesWriter.write`.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np

from neofoam.postprocess.writers.vtkhdf import MeshExport

#: ``"process"``: a file per process; ``"node"``: per node (shared memory); ``k``: per k ranks
Group = Union[Literal["node", "process"], int]

_MESH = (  # dataset, MeshExport method, dtype
    ("Points", "points", np.float64),
    ("Connectivity", "connectivity", np.int64),
    ("Offsets", "offsets", np.int64),
    ("Types", "cell_types", np.uint8),
)


def _h5py() -> Any:
    import h5py  # noqa: PLC0415 - optional, only a run that writes VTKHDF needs it

    return h5py


def _mesh_arrays(mesh: MeshExport) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(getattr(mesh, method)(), dtype=t) for name, method, t in _MESH}
    arrays["Points"] = arrays["Points"].reshape(-1, 3)
    return arrays


def _appendable(group: Any, name: str, data: np.ndarray) -> None:
    """Append ``data`` (rows) to the resizable dataset ``group[name]``, created on first use."""
    if name not in group:
        chunks = (max(1, min(len(data), 65536)), *data.shape[1:])
        group.create_dataset(name, data=data, maxshape=(None, *data.shape[1:]), chunks=chunks)
        return
    dataset = group[name]
    start = dataset.shape[0]
    dataset.resize(start + data.shape[0], axis=0)
    dataset[start:] = data


def _split(comm: Any, group: Group) -> Any:
    """The communicator of this rank's group, ranks in world order."""
    if group == "node":
        from mpi4py import MPI  # noqa: PLC0415 - only with a communicator

        return comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)
    if group == "process":
        return comm.Split(comm.rank, key=0)
    if isinstance(group, int) and not isinstance(group, bool) and group > 0:
        return comm.Split(comm.rank // group, key=comm.rank)
    raise ValueError(f"group must be 'node', 'process' or a positive int, not {group!r}")


class VtkHdfFilesWriter:
    """Write one static mesh and its point and cell data over time, a file per process group.

    The constructor and :meth:`write` match :class:`~.vtkhdf.VtkHdfWriter`'s and
    are collective over ``comm`` (``None``: serial, one group file). ``path`` is the
    index to open in ParaView; the group files go to ``path.parent / path.stem``.
    Its parts are in group order, the members of a group in rank order: rank order
    unless ``group="node"`` places ranks round robin. Needs h5py (serial is enough).
    Example::

        out = VtkHdfFilesWriter(path, mesh, comm, group="node")
        out.write(0.1, point_data={"T": T}, cell_data={"p": p})
    """

    def __init__(
        self, path: Path, mesh: MeshExport, comm: Any = None, *, group: Group = "node"
    ) -> None:
        self.path = Path(path)
        self._comm = comm
        if comm is not None:
            gc.collect()  # every rank at one point, see VtkHdfWriter
            self._group = _split(comm, group)
            leader = self._group.bcast(comm.rank, root=0)
            leaders = sorted(set(comm.allgather(leader)))
            self._index = leaders.index(leader)
        else:
            self._group = None
            self._index = 0
        directory = self.path.parent / self.path.stem
        self.file = directory / f"{self.path.stem}_{self._index}.vtkhdf"
        arrays = _mesh_arrays(mesh)
        parts = self._gather(arrays)
        counts = (len(arrays["Points"]), len(arrays["Types"]), len(arrays["Connectivity"]))
        # per group (in group order), the counts (points, cells, connectivity ids) per part
        mine = [counts] if self._group is None else self._group.gather(counts, root=0)
        groups = [mine] if comm is None else comm.gather(mine if parts else None, root=0)
        self._groups = [g for g in groups or [] if g is not None]
        self._sizes = {
            "Point": [len(part["Points"]) for part in parts],
            "Cell": [len(part["Types"]) for part in parts],
        }
        self._fields: dict[str, dict[str, tuple[int, ...]]] = {"Point": {}, "Cell": {}}
        self._times: list[float] = []
        if parts:
            directory.mkdir(parents=True, exist_ok=True)
            self._create(parts)
        self._barrier()
        if self._groups:
            self._write_index()

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
        self._barrier()
        if self._groups:
            self._times.append(float(time))
            for kind in ("Point", "Cell"):
                self._fields[kind] = {
                    name: np.shape(values)[1:] for name, values in data[kind].items()
                }
            self._write_index()

    def _gather(self, piece: dict[str, Any]) -> list[dict[str, Any]]:
        """The group's pieces on its leader in rank order, ``[]`` on the others."""
        if self._group is None:
            return [piece]
        return self._group.gather(piece, root=0) or []

    def _barrier(self) -> None:
        if self._comm is not None:
            self._comm.Barrier()

    def _create(self, parts: list[dict[str, np.ndarray]]) -> None:
        with _h5py().File(self.file, "w") as f:
            root = _root(f)
            root.create_dataset("NumberOfPoints", data=self._sizes["Point"], dtype=np.int64)
            root.create_dataset("NumberOfCells", data=self._sizes["Cell"], dtype=np.int64)
            ids = [len(part["Connectivity"]) for part in parts]
            root.create_dataset("NumberOfConnectivityIds", data=ids, dtype=np.int64)
            for name, _, _ in _MESH:
                root.create_dataset(name, data=np.concatenate([part[name] for part in parts]))
            root.create_group("PointData")
            root.create_group("CellData")
            steps = root.create_group("Steps")
            steps.attrs["NSteps"] = 0
            steps.create_group("PointDataOffsets")
            steps.create_group("CellDataOffsets")

    def _append(self, time: float, parts: list[dict[str, dict[str, np.ndarray]]]) -> None:
        with _h5py().File(self.file, "a") as f:
            root = f["VTKHDF"]
            steps = root["Steps"]
            _appendable(steps, "Values", np.array([time], dtype=float))
            _appendable(steps, "NumberOfParts", np.array([len(parts)], dtype=np.int64))
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

    def _write_index(self) -> None:
        """Rewrite the index over the group files (rank 0): a temporary file, then renamed."""
        h5py = _h5py()
        sources = [
            f"{self.path.stem}/{self.path.stem}_{g}.vtkhdf" for g in range(len(self._groups))
        ]
        counts = np.array([c for g in self._groups for c in g], dtype=np.int64).reshape(-1, 3)
        # per group: rows of its mesh arrays and of its point and cell data per step
        rows = {
            "Points": [sum(c[0] for c in g) for g in self._groups],
            "Types": [sum(c[1] for c in g) for g in self._groups],
            "Connectivity": [sum(c[2] for c in g) for g in self._groups],
        }
        rows["Offsets"] = [r + len(g) for r, g in zip(rows["Types"], self._groups)]
        n_steps = len(self._times)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with h5py.File(temporary, "w") as f:
            root = _root(f)
            for i, name in enumerate(
                ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds")
            ):
                root.create_dataset(name, data=counts[:, i])
            for name, _, dtype in _MESH:
                points_tail: tuple[int, ...] = (3,) if name == "Points" else ()
                _virtual(root, name, sources, rows[name], points_tail, dtype, 1)
            steps = root.create_group("Steps")
            steps.attrs["NSteps"] = n_steps
            if n_steps:
                steps.create_dataset("Values", data=np.array(self._times))
                steps.create_dataset("NumberOfParts", data=np.full(n_steps, len(counts)))
                for name in ("PartOffsets", "PointOffsets", "CellOffsets"):
                    steps.create_dataset(name, data=np.zeros(n_steps, dtype=np.int64))
                steps.create_dataset("ConnectivityIdOffsets", data=np.zeros((n_steps, 1), np.int64))
            for kind, rows_per_step in (("Point", rows["Points"]), ("Cell", rows["Types"])):
                data = root.create_group(f"{kind}Data")
                offsets = steps.create_group(f"{kind}DataOffsets")
                for name, tail in self._fields[kind].items():
                    _virtual(
                        data, f"{kind}Data/{name}", sources, rows_per_step, tail, float, n_steps
                    )
                    offsets.create_dataset(name, data=np.arange(n_steps) * sum(rows_per_step))
        os.replace(temporary, self.path)


def _root(f: Any) -> Any:
    root = f.create_group("VTKHDF")
    root.attrs["Version"] = (2, 1)
    kind = "UnstructuredGrid"
    root.attrs.create("Type", np.bytes_(kind), dtype=f"S{len(kind)}")
    return root


def _virtual(
    parent: Any,
    path: str,
    sources: list[str],
    rows: list[int],
    tail: tuple[int, ...],
    dtype: Any,
    n_steps: int,
) -> None:
    """``parent[name]``: ``n_steps`` blocks of the groups' ``rows`` each, block ``i`` of group
    ``g`` read from rows ``i * rows[g]`` on of ``VTKHDF/<path>`` in ``sources[g]``."""
    h5py = _h5py()
    name = path.rsplit("/", 1)[-1]
    total = sum(rows)
    shape = (n_steps * total, *tail)
    if total == 0 or n_steps == 0:
        parent.create_dataset(name, shape=shape, dtype=dtype)
        return
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
    start = 0
    for source, n in zip(sources, rows):
        if n:
            source_shape = (n_steps * n, *tail)
            virtual = h5py.h5s.create_simple(shape)
            # every step's block of this group: n rows at stride total, starting at start
            virtual.select_hyperslab(
                (start, *(0 for _ in tail)), (n_steps, *(1 for _ in tail)),
                stride=(total, *(1 for _ in tail)), block=(n, *tail),
            )  # fmt: skip
            src = h5py.h5s.create_simple(source_shape)
            layout.dcpl.set_virtual(virtual, source.encode(), f"VTKHDF/{path}".encode(), src)
        start += n
    parent.create_virtual_dataset(name, layout)
