# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun solver package: one solver per region, stepped by one time loop."""

from .multiRun import multiRun, run

__all__ = ["multiRun", "run"]
