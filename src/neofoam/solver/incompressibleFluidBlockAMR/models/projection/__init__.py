# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""blockAMR projection (pressure-velocity) model package.

Only the Chorin fractional-step projection is wired up (MAC + nodal projection
via the ``neon.blockamr`` DSL).
"""

from .base import ProjectionAlgorithm
from . import chorinProjection as _chorin  # noqa: F401

__all__ = ["ProjectionAlgorithm"]
