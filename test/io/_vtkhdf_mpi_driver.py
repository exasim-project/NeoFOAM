# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run per rank under ``mpirun``: ``argv[2]`` names each rank's part (default ``hex,tet``)."""

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from neofoam.postprocess.writers.vtkhdf import UnstructuredMesh, VtkHdfWriter

CUBE = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
PIECES = {
    "hex": UnstructuredMesh(np.array(CUBE), np.arange(8), np.array([0, 8]), np.array([12])),
    "tet": UnstructuredMesh(
        np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0.5, 0.5, 2]]),
        np.arange(4),
        np.array([0, 4]),
        np.array([10]),
    ),
    "empty": UnstructuredMesh(
        np.zeros((0, 3)), np.zeros(0, int), np.array([0]), np.zeros(0, np.uint8)
    ),
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
piece = PIECES[(sys.argv[2] if len(sys.argv) > 2 else "hex,tet").split(",")[rank]]
out = VtkHdfWriter(Path(sys.argv[1]), piece, comm)
for step, time in enumerate((0.1, 0.2)):
    n_points = len(piece.points())
    out.write(
        time,
        point_data={"T": 100.0 * rank + 10.0 * step + np.arange(n_points)},
        cell_data={"p": np.full(len(piece.cell_types()), step + 0.5 * rank)},
    )
