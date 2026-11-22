# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import pytest
from pydantic import ValidationError

from neofoam.telemetry import MpiInfo, TelemetrySettings


def test_settings_defaults() -> None:
    settings = TelemetrySettings()
    assert settings.enabled is True
    assert settings.directory == "telemetry"
    assert settings.summary is True
    assert settings.service_name == "neofoam"


def test_settings_rejects_unknown_types() -> None:
    with pytest.raises(ValidationError):
        TelemetrySettings(enabled="not-a-bool")  # type: ignore[arg-type]


def test_mpi_info_defaults_to_serial() -> None:
    mpi = MpiInfo()
    assert mpi.rank == 0
    assert mpi.size == 1
    assert mpi.par_run is False


def test_mpi_info_holds_parallel_values() -> None:
    mpi = MpiInfo(rank=3, size=8, par_run=True)
    assert mpi.rank == 3
    assert mpi.size == 8
    assert mpi.par_run is True
