# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity model package entrypoint (minimal port).

Only the PIMPLE algorithm is wired up; SIMPLE / PISO from the source
branch are intentionally omitted for the minimal version.
"""

from .base import PressureVelocityAlgorithm
from . import pimpleAlgorithm as _pimple  # noqa: F401

__all__ = ["PressureVelocityAlgorithm"]
