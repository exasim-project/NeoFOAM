# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The incompressibleFluid spec, importable by its load step without an import cycle."""

from neofoam.framework.solver import Solver

incompressibleFluid = Solver("incompressibleFluid")
