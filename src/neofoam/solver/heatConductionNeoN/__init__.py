# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""heatConductionNeoN: heat conduction on NeoN, with coupled regions in ``multiRun``."""

from .heatConductionNeoN import heatConductionNeoN, run

__all__ = ["heatConductionNeoN", "run"]
