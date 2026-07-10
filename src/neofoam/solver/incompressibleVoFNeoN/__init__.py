# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoFNeoN solver package.

Framework port of the NeoN-backed ``neoInterFoam`` VoF solver: the same
SolverSpec / ModelSpec / StagedInit composition as ``incompressibleVoF``, with
the NeoN bindings (``neon._neon`` / ``neofoam.neofoam_bindings``) as the
backend. interFoam-style two-phase VoF — MULES phase-fraction advection, a
density-weighted momentum predictor with gravity + surface tension, and a
buoyant ``p_rgh`` pressure correction. Laminar only (no turbulence model).
"""

from .config_schema import config_classes
from .incompressibleVoFNeoN import incompressibleVoFNeoN, run

__all__ = ["incompressibleVoFNeoN", "run", "config_classes"]
