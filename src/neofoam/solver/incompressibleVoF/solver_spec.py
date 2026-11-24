# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The incompressibleVoF spec, importable by its load step without an import cycle."""

from neofoam.framework.solver import Solver

incompressibleVoF = Solver("incompressibleVoF")
