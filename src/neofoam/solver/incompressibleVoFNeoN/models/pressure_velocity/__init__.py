# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoN VoF pressure-velocity model package entrypoint.

VoF always uses PIMPLE (interFoam does not support SIMPLE or PISO).
"""

from .base import PressureVelocityAlgorithmNeoN
from . import pimpleAlgorithm as _pimple  # noqa: F401

__all__ = ["PressureVelocityAlgorithmNeoN"]
