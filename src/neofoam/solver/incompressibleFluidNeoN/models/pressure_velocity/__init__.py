# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoN pressure-velocity model package entrypoint.

PIMPLE (transient) and SIMPLE (steady state) are wired up; PISO falls back to PIMPLE.
"""

from . import pimpleAlgorithm as _pimple  # noqa: F401
from .base import PressureVelocityAlgorithmNeoN

__all__ = ["PressureVelocityAlgorithmNeoN"]
