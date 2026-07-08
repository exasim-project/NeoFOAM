# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoN pressure-velocity model package entrypoint.

Only the PIMPLE algorithm is wired up; SIMPLE / PISO fall back to PIMPLE.
"""

from .base import PressureVelocityAlgorithmNeoN
from . import pimpleAlgorithm as _pimple  # noqa: F401

__all__ = ["PressureVelocityAlgorithmNeoN"]
