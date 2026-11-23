# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the VTKHDF writer of a file per process group (``writers.vtkhdf_files``).

Serial: one group file and the index over it read as :class:`VtkHdfWriter`'s file.
Under ``mpirun -np 3`` (``_vtkhdf_files_mpi_driver.py``; rank 0 a hexahedron, rank 1
empty, rank 2 a tetrahedron, two steps of ``T = 100 * rank + 10 * step + point``
and a cell vector ``U = step + 0.5 * rank``) for each grouping: the group files and
their parts, and the index equal, dataset by dataset, to the file
:class:`VtkHdfWriter` writes of the same data, and read in VTK at the second step.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _vtkhdf_mpirun import mpirun, needs_mpi

from neofoam.postprocess.writers import VtkHdfFilesWriter, VtkHdfWriter, vtkhdf_writer
from neofoam.postprocess.writers.vtkhdf import UnstructuredMesh

h5py = pytest.importorskip("h5py")

DRIVER = Path(__file__).parent / "_vtkhdf_files_mpi_driver.py"
T_STEP_1 = [10, 11, 12, 13, 14, 15, 16, 17, 210, 211, 212, 213]
#: per grouping, the points per part of each group file
GROUPS: dict[str, list[list[int]]] = {
    "process": [[8], [0], [4]],
    "2": [[8, 0], [4]],
    "node": [[8, 0, 4]],
}
HEX_AND_TET = UnstructuredMesh(
    np.array([*np.ndindex(2, 2, 2), (0.5, 0.5, 2)], dtype=float),
    np.array([0, 1, 3, 2, 4, 5, 7, 6, 4, 5, 6, 8]),
    np.array([0, 8, 12]),
    np.array([12, 10]),
)


def _datasets(f: Any) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    f.visititems(
        lambda name, item: found.update({name: item[()]}) if hasattr(item, "shape") else None
    )
    return found


def _assert_same_file(path: Path, reference: Path) -> None:
    with h5py.File(path, "r") as f, h5py.File(reference, "r") as g:
        ours, theirs = _datasets(f), _datasets(g)
        assert sorted(ours) == sorted(theirs)
        for name, values in theirs.items():
            np.testing.assert_array_equal(ours[name], values, err_msg=name)
        assert f["VTKHDF/Steps"].attrs["NSteps"] == g["VTKHDF/Steps"].attrs["NSteps"]


def _write(out: Any) -> None:
    for step, time in enumerate((0.1, 0.2)):
        U = np.array([[step, 0, 0], [step, 1, 0]], dtype=float)
        out.write(time, point_data={"T": 10.0 * step + np.arange(9.0)}, cell_data={"U": U})


def test_files_writer_serial_index_equals_the_one_file(tmp_path: Path) -> None:
    out = vtkhdf_writer(tmp_path / "a.vtkhdf", HEX_AND_TET, kind="files")
    _write(out)
    _write(VtkHdfWriter(tmp_path / "one.vtkhdf", HEX_AND_TET))

    assert isinstance(out, VtkHdfFilesWriter)
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["a_0.vtkhdf"]
    _assert_same_file(tmp_path / "a.vtkhdf", tmp_path / "one.vtkhdf")
    _assert_same_file(tmp_path / "a" / "a_0.vtkhdf", tmp_path / "one.vtkhdf")


def test_files_writer_index_is_valid_before_the_first_step(tmp_path: Path) -> None:
    VtkHdfFilesWriter(tmp_path / "a.vtkhdf", HEX_AND_TET)

    with h5py.File(tmp_path / "a.vtkhdf", "r") as f:
        assert f["VTKHDF/Steps"].attrs["NSteps"] == 0
        assert list(f["VTKHDF/NumberOfPoints"]) == [9]
        assert list(f["VTKHDF/Types"]) == [12, 10]


def test_files_writer_rejects_a_field_of_the_wrong_length(tmp_path: Path) -> None:
    out = VtkHdfFilesWriter(tmp_path / "a.vtkhdf", HEX_AND_TET)

    with pytest.raises(ValueError, match="point data T"):
        out.write(0.1, point_data={"T": np.zeros(8)})


def test_vtkhdf_writer_selects_parallel_by_default(tmp_path: Path) -> None:
    assert isinstance(vtkhdf_writer(tmp_path / "a.vtkhdf", HEX_AND_TET), VtkHdfWriter)
    with pytest.raises(ValueError, match="kind"):
        vtkhdf_writer(tmp_path / "b.vtkhdf", HEX_AND_TET, kind="gather")  # type: ignore[arg-type]


def test_files_writer_index_reads_in_vtk_from_elsewhere(tmp_path: Path) -> None:
    vtk = pytest.importorskip("vtk")
    vtk_to_numpy = pytest.importorskip("vtk.util.numpy_support").vtk_to_numpy
    _write(VtkHdfFilesWriter(tmp_path / "run" / "a.vtkhdf", HEX_AND_TET))
    (tmp_path / "run").rename(tmp_path / "moved")  # the sources are relative to the index

    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(tmp_path / "moved" / "a.vtkhdf"))
    reader.UpdateInformation()
    reader.UpdateTimeStep(0.2)
    grid = reader.GetOutput()

    assert (grid.GetNumberOfPoints(), grid.GetNumberOfCells()) == (9, 2)
    np.testing.assert_array_equal(
        vtk_to_numpy(grid.GetPointData().GetArray("T")), 10 + np.arange(9)
    )


@pytest.fixture(scope="module", params=list(GROUPS))
def written(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Path]:
    directory = tmp_path_factory.mktemp(f"vtkhdf_files_{request.param}")
    command = [
        str(mpirun()),
        "-np",
        "3",
        sys.executable,
        str(DRIVER),
        str(directory),
        request.param,
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return request.param, directory


@needs_mpi
def test_files_writer_mpi_writes_a_file_per_group(written: tuple[str, Path]) -> None:
    group, directory = written
    expected = GROUPS[group]
    files = sorted((directory / "files").iterdir())

    assert [p.name for p in files] == [f"files_{g}.vtkhdf" for g in range(len(expected))]
    for path, points in zip(files, expected):
        with h5py.File(path, "r") as f:
            assert list(f["VTKHDF/NumberOfPoints"]) == points
            assert list(f["VTKHDF/Steps/NumberOfParts"]) == [len(points)] * 2


@needs_mpi
def test_files_writer_mpi_index_equals_the_one_file(written: tuple[str, Path]) -> None:
    _, directory = written

    _assert_same_file(directory / "files.vtkhdf", directory / "one.vtkhdf")


@needs_mpi
def test_files_writer_mpi_index_reads_in_vtk_at_the_second_step(written: tuple[str, Path]) -> None:
    vtk = pytest.importorskip("vtk")
    vtk_to_numpy = pytest.importorskip("vtk.util.numpy_support").vtk_to_numpy
    _, directory = written

    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(directory / "files.vtkhdf"))
    reader.UpdateInformation()
    reader.UpdateTimeStep(0.2)
    parts = reader.GetOutput()  # a vtkPartitionedDataSet, one partition per part

    assert parts.GetNumberOfPartitions() == 3
    assert [parts.GetPartition(i).GetNumberOfPoints() for i in range(3)] == [8, 0, 4]
    T = [parts.GetPartition(i).GetPointData().GetArray("T") for i in (0, 2)]
    np.testing.assert_array_equal(np.concatenate([vtk_to_numpy(a) for a in T]), T_STEP_1)
    U = vtk_to_numpy(parts.GetPartition(2).GetCellData().GetArray("U"))
    np.testing.assert_array_equal(U, [[2.0, 2.0, 2.0]])
