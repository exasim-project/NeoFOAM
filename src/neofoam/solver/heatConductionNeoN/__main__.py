# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``python -m neofoam.solver.heatConductionNeoN`` runs the case in the working directory."""

import sys

from .heatConductionNeoN import run

run(sys.argv)
