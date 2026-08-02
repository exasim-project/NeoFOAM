# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""blockAMR projection (pressure-velocity) model package.

Only the Chorin fractional-step projection is wired up (MAC + nodal projection
via the ``blockamr`` DSL).
"""

from . import chorinProjection as _chorin  # noqa: F401
from .base import ProjectionAlgorithm

__all__ = ["ProjectionAlgorithm"]
