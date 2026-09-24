# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Plain-data settings for the telemetry shim (no OpenTelemetry imports)."""

from pydantic import BaseModel


class MpiInfo(BaseModel):
    """Identity of this process in an MPI run (serial defaults)."""

    rank: int = 0
    size: int = 1
    par_run: bool = False


class TelemetrySettings(BaseModel):
    """How tracing behaves once configured.

    ``directory`` is resolved relative to the case directory; each rank
    writes ``rank<N>.spans.jsonl`` (and ``rank<N>.summary.json`` when
    ``summary`` is on) into it.
    """

    enabled: bool = True
    directory: str = "telemetry"
    summary: bool = True
    service_name: str = "neofoam"
