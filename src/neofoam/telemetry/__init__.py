# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Opt-in OpenTelemetry performance tracing (optional ``telemetry`` extra).

Safe to import without OpenTelemetry installed: only :func:`configure`
touches the SDK (via a lazy import) and raises
:class:`TelemetryNotInstalledError` when the extra is missing.
"""

from .api import (
    TelemetryNotInstalledError,
    configure,
    instrument,
    is_active,
    shutdown,
    span,
)
from .report import (
    load_spans,
    load_summaries,
    plot_summary,
    to_chrome_trace,
    write_chrome_trace,
    write_summary_plot,
)
from .settings import MpiInfo, TelemetrySettings

__all__ = [
    "MpiInfo",
    "TelemetryNotInstalledError",
    "TelemetrySettings",
    "configure",
    "instrument",
    "is_active",
    "load_spans",
    "load_summaries",
    "plot_summary",
    "shutdown",
    "span",
    "to_chrome_trace",
    "write_chrome_trace",
    "write_summary_plot",
]
