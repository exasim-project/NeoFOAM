# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the VTKHDF writer under ``mpirun``: one part per rank in one file.

``_vtkhdf_mpi_driver.py`` runs on two ranks, rank 0 a hexahedron (8 points),
rank 1 a tetrahedron (4 points, its own numbering), two steps of a point field
``T = 100 * rank + 10 * step + local point index`` and a cell field
``p = step + 0.5 * rank``. Every rank writes its own part collectively, so the
file has two parts whose mesh arrays and per-step data follow each other in rank
order. A third layout puts a rank without cells between them. The ranks run in a
subprocess: the test process has no MPI of its own, and its timeout turns a hang
in a collective into a failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _vtkhdf_mpirun import mpirun, needs_mpi

h5py = pytest.importorskip("h5py")

DRIVER = Path(__file__).parent / "_vtkhdf_mpi_driver.py"
MISMATCH_DRIVER = Path(__file__).parent / "_vtkhdf_mpi_mismatch_driver.py"
T_STEP_1 = [10, 11, 12, 13, 14, 15, 16, 17, 110, 111, 112, 113]

pytestmark = needs_mpi


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("vtkhdf_mpi") / "hexAndTet.vtkhdf"
    command = [str(mpirun()), "-np", "2", sys.executable, str(DRIVER), str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return path


def test_vtkhdf_writer_mpi_writes_one_part_per_rank(written: Path) -> None:
    with h5py.File(written, "r") as f:
        root = f["VTKHDF"]

        assert list(root["NumberOfPoints"]) == [8, 4]
        assert list(root["NumberOfCells"]) == [1, 1]
        assert list(root["NumberOfConnectivityIds"]) == [8, 4]
        assert list(root["Offsets"]) == [0, 8, 0, 4]
        assert list(root["Connectivity"]) == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
        assert list(root["Types"]) == [12, 10]
        assert list(root["Steps/NumberOfParts"]) == [2, 2]


def test_vtkhdf_writer_mpi_step_data_follows_rank_order(written: Path) -> None:
    with h5py.File(written, "r") as f:
        root = f["VTKHDF"]
        point = root["Steps/PointDataOffsets/T"][1]
        cell = root["Steps/CellDataOffsets/p"][1]

        np.testing.assert_array_equal(root["PointData/T"][point : point + 12], T_STEP_1)
        np.testing.assert_array_equal(root["CellData/p"][cell : cell + 2], [1.0, 1.5])


def test_vtkhdf_writer_mpi_parts_read_in_vtk_at_a_later_step(written: Path) -> None:
    vtk = pytest.importorskip("vtk")
    vtk_to_numpy = pytest.importorskip("vtk.util.numpy_support").vtk_to_numpy

    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(written))
    reader.UpdateInformation()
    reader.UpdateTimeStep(0.2)
    parts = reader.GetOutput()  # a vtkPartitionedDataSet, one partition per part

    assert (parts.GetNumberOfPartitions(), parts.GetNumberOfPoints()) == (2, 12)
    assert parts.GetNumberOfCells() == 2
    T = [vtk_to_numpy(parts.GetPartition(i).GetPointData().GetArray("T")) for i in range(2)]
    np.testing.assert_array_equal(np.concatenate(T), T_STEP_1)


def test_vtkhdf_writer_mpi_keeps_a_rank_without_cells_as_an_empty_part(tmp_path: Path) -> None:
    path = tmp_path / "hexEmptyTet.vtkhdf"
    command = [str(mpirun()), "-np", "3", sys.executable, str(DRIVER), str(path), "hex,empty,tet"]

    subprocess.run(command, check=True, capture_output=True, timeout=120)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert list(root["NumberOfPoints"]) == [8, 0, 4]
        assert list(root["NumberOfCells"]) == [1, 0, 1]
        assert list(root["NumberOfConnectivityIds"]) == [8, 0, 4]
        assert list(root["Offsets"]) == [0, 8, 0, 0, 4]
        assert list(root["Steps/NumberOfParts"]) == [3, 3]
        assert list(root["Steps/PointDataOffsets/T"]) == [0, 12]
        expected_T = [10, 11, 12, 13, 14, 15, 16, 17, 210, 211, 212, 213]
        np.testing.assert_array_equal(root["PointData/T"][12:24], expected_T)
        np.testing.assert_array_equal(root["CellData/p"][2:4], [1.0, 2.0])


def test_vtkhdf_writer_mpi_raises_on_every_rank_for_different_field_names(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.vtkhdf"
    command = [str(mpirun()), "-np", "2", sys.executable, str(MISMATCH_DRIVER), str(path)]

    done = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)

    assert done.stdout.count("the ranks pass different fields") == 2
