# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun: one solver per region, each on its own thread, coupled through a comm channel."""

from .multiRun import multiRun, run

__all__ = ["multiRun", "run"]
