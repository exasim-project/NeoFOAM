# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""MPI identity via OpenFOAM's Pstream (the only pybFoam touchpoint here).

Imported lazily by :func:`neofoam.telemetry.configure` so importing the
telemetry shim never pulls in pybFoam.
"""

from pybFoam import Pstream

from .settings import MpiInfo


def current_mpi_info() -> MpiInfo:
    """Rank/size of this process as OpenFOAM sees it (serial: 0 of 1)."""
    return MpiInfo(
        rank=Pstream.myProcNo(), size=Pstream.nProcs(), par_run=Pstream.parRun()
    )
