# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run per rank under ``mpirun -np 3``: a hexahedron, nothing, a tetrahedron.

``argv``: the directory and the group (``process``, ``node`` or an int). It writes
``files.vtkhdf`` with :class:`VtkHdfFilesWriter` and, as the reference,
``one.vtkhdf`` with :class:`VtkHdfWriter`, two steps of the same data.
"""

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from neofoam.postprocess.writers import vtkhdf_writer
from neofoam.postprocess.writers.vtkhdf import UnstructuredMesh

CUBE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
PIECES = [
    UnstructuredMesh(np.array(CUBE), np.arange(8), np.array([0, 8]), np.array([12])),
    UnstructuredMesh(np.zeros((0, 3)), np.zeros(0), np.array([0]), np.zeros(0)),
    UnstructuredMesh(
        np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0.5, 0.5, 2]]),
        np.arange(4),
        np.array([0, 4]),
        np.array([10]),
    ),
]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
piece = PIECES[rank]
directory, group = Path(sys.argv[1]), sys.argv[2]
writers = [
    vtkhdf_writer(
        directory / "files.vtkhdf", piece, comm, "files", int(group) if group.isdigit() else group
    ),  # type: ignore[arg-type]  # noqa: E501
    vtkhdf_writer(directory / "one.vtkhdf", piece, comm),
]
n_points, n_cells = len(piece.points()), len(piece.cell_types())
for step, time in enumerate((0.1, 0.2)):
    for out in writers:
        out.write(
            time,
            point_data={"T": 100.0 * rank + 10.0 * step + np.arange(n_points)},
            cell_data={"U": np.full((n_cells, 3), step + 0.5 * rank)},
        )
