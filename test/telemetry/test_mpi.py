# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors


from neofoam.telemetry.mpi import current_mpi_info  # noqa: E402


def test_current_mpi_info_serial() -> None:
    mpi = current_mpi_info()
    assert mpi.rank == 0
    assert mpi.size == 1
    assert mpi.par_run is False
