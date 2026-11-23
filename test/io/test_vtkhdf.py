# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the transient VTKHDF writer (``neofoam.postprocess.writers.vtkhdf``), serial.

The mesh mixes cell types: a unit-cube hexahedron (points 0-7) and a tetrahedron
on its top face (points 4, 5, 6 and the apex 8), so the offsets are not a constant
stride. Three steps write a point field ``T = 10 * step + point index`` and a cell
vector ``U = (step, cell, 0)``: every value names its step, so a wrong offset shows.
The file is read back with h5py (the layout) and, where VTK is installed, with
VTK's own ``vtkHDFReader`` (that ParaView reads it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neofoam.postprocess.writers.vtkhdf import UnstructuredMesh, VtkHdfWriter

h5py = pytest.importorskip("h5py")

HEX_AND_TET = UnstructuredMesh(
    points=np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 2],
        ]
    ),
    connectivity=np.array([0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 8]),
    offsets=np.array([0, 8, 12]),
    cell_types=np.array([12, 10]),  # VTK_HEXAHEDRON, VTK_TETRA
)
TIMES = (0.1, 0.2, 0.3)


def _point_T(step: int) -> np.ndarray:
    return 10.0 * step + np.arange(9.0)


def _cell_U(step: int) -> np.ndarray:
    return np.array([[step, 0, 0], [step, 1, 0]], dtype=float)


@pytest.fixture
def written(tmp_path: Path) -> Path:
    path = tmp_path / "hexAndTet.vtkhdf"
    out = VtkHdfWriter(path, HEX_AND_TET)
    for step, time in enumerate(TIMES):
        out.write(time, point_data={"T": _point_T(step)}, cell_data={"U": _cell_U(step)})
    return path


def test_vtkhdf_writer_writes_the_mixed_mesh_as_one_part(written: Path) -> None:
    with h5py.File(written, "r") as f:
        root = f["VTKHDF"]

        assert root.attrs["Type"] == b"UnstructuredGrid"
        assert tuple(root.attrs["Version"]) == (2, 1)
        assert list(root["NumberOfPoints"]) == [9]
        assert list(root["NumberOfCells"]) == [2]
        assert list(root["NumberOfConnectivityIds"]) == [12]
        assert list(root["Offsets"]) == [0, 8, 12]
        assert list(root["Types"]) == [12, 10]
        assert list(root["Connectivity"]) == [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 8]


def test_vtkhdf_writer_steps_share_the_mesh(written: Path) -> None:
    with h5py.File(written, "r") as f:
        steps = f["VTKHDF/Steps"]

        assert steps.attrs["NSteps"] == 3
        np.testing.assert_array_equal(steps["Values"], TIMES)
        assert list(steps["NumberOfParts"]) == [1, 1, 1]
        for name in ("PartOffsets", "PointOffsets", "CellOffsets", "ConnectivityIdOffsets"):
            assert not np.any(steps[name][()]), name


@pytest.mark.parametrize("step", [0, 1, 2])
def test_vtkhdf_writer_data_offsets_find_each_steps_values(written: Path, step: int) -> None:
    with h5py.File(written, "r") as f:
        root = f["VTKHDF"]
        point = root["Steps/PointDataOffsets/T"][step]
        cell = root["Steps/CellDataOffsets/U"][step]

        np.testing.assert_array_equal(root["PointData/T"][point : point + 9], _point_T(step))
        np.testing.assert_array_equal(root["CellData/U"][cell : cell + 2], _cell_U(step))


def test_vtkhdf_writer_rejects_a_field_of_the_wrong_length(tmp_path: Path) -> None:
    out = VtkHdfWriter(tmp_path / "hexAndTet.vtkhdf", HEX_AND_TET)

    with pytest.raises(ValueError, match="point data T"):
        out.write(0.1, point_data={"T": np.zeros(8)})


def test_vtkhdf_writer_file_reads_in_vtk_at_a_later_step(written: Path) -> None:
    vtk = pytest.importorskip("vtk")
    vtk_to_numpy = pytest.importorskip("vtk.util.numpy_support").vtk_to_numpy

    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(written))
    reader.UpdateInformation()
    reader.UpdateTimeStep(TIMES[2])
    grid = reader.GetOutput()

    assert (grid.GetNumberOfPoints(), grid.GetNumberOfCells()) == (9, 2)
    assert [grid.GetCellType(cell) for cell in range(2)] == [12, 10]
    np.testing.assert_array_equal(vtk_to_numpy(grid.GetPointData().GetArray("T")), _point_T(2))
    np.testing.assert_array_equal(vtk_to_numpy(grid.GetCellData().GetArray("U")), _cell_U(2))
