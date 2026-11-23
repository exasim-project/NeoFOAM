# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The writers shipped with NeoFOAM: the CSV table output and the VTKHDF field output.

Importing this subpackage registers the table writers, so a case file can select
one by its ``type`` string. A table writer is the only part of the package besides
the sources and the reductions that touches pybFoam — to ask which rank owns the
output files. :class:`VtkHdfWriter` is not a table writer: a backend constructs it
directly, without pybFoam, and passes an mpi4py communicator when decomposed.
"""

from neofoam.postprocess.writers.csv import CsvWriter
from neofoam.postprocess.writers.vtkhdf import MeshExport, UnstructuredMesh, VtkHdfWriter
from neofoam.postprocess.writers.writer import TableWriter, table_headers, table_rows

__all__ = [
    "TableWriter",
    "CsvWriter",
    "table_headers",
    "table_rows",
    "MeshExport",
    "UnstructuredMesh",
    "VtkHdfWriter",
]
