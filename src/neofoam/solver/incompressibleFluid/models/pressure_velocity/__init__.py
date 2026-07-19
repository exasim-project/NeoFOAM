# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity model package entrypoint.

PIMPLE (transient) and SIMPLE (steady state) are wired up; PISO from
the source branch is intentionally omitted for the minimal version.
"""

from .base import PressureVelocityAlgorithm
from . import pimpleAlgorithm as _pimple  # noqa: F401
from . import simpleAlgorithm as _simple  # noqa: F401

__all__ = ["PressureVelocityAlgorithm"]
