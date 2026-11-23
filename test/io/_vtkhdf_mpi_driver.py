# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run per rank under ``mpirun -np 2``: rank 0 writes a hexahedron, rank 1 a tetrahedron."""

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from neofoam.postprocess.writers.vtkhdf import UnstructuredMesh, VtkHdfWriter

CUBE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
PIECES = [
    UnstructuredMesh(np.array(CUBE), np.arange(8), np.array([0, 8]), np.array([12])),
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
out = VtkHdfWriter(Path(sys.argv[1]), piece, comm)
for step, time in enumerate((0.1, 0.2)):
    n_points = len(piece.points())
    out.write(
        time,
        point_data={"T": 100.0 * rank + 10.0 * step + np.arange(n_points)},
        cell_data={"p": np.array([step + 0.5 * rank])},
    )
