# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run per rank under ``mpirun -np 2``: the ranks pass different field names to one write."""

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from neofoam.postprocess.writers.vtkhdf import UnstructuredMesh, VtkHdfWriter

comm = MPI.COMM_WORLD
mesh = UnstructuredMesh(
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    np.arange(4),
    np.array([0, 4]),
    np.array([10]),
)
out = VtkHdfWriter(Path(sys.argv[1]), mesh, comm)
try:
    out.write(0.1, point_data={"T" if comm.Get_rank() == 0 else "q": np.zeros(4)})
except ValueError as error:
    print(f"rank {comm.Get_rank()} raised: {error}", flush=True)
    sys.exit(0)
sys.exit(f"rank {comm.Get_rank()}: write accepted different field names")
