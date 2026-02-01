# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SimpleSolver optional models.

Import all available model implementations.
"""

from .base import SimpleSolverModel
from .boussinesq import BoussinesqModel

__all__ = [
    "SimpleSolverModel",
    "BoussinesqModel",
]
