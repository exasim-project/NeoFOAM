# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoN pressure-velocity model package entrypoint.

Only the PIMPLE algorithm is wired up; SIMPLE / PISO fall back to PIMPLE.
"""

from . import pimpleAlgorithm as _pimple  # noqa: F401
from .base import PressureVelocityAlgorithmNeoN

__all__ = ["PressureVelocityAlgorithmNeoN"]
