# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleFluidNeoN.

Pydantic ``BaseConfig`` for ``system/controlDict``, loaded via
``@IOStrategy(OF(...))`` — it feeds validation of the time-stepping controls
before the solver opens any C++ runtime. ``constant/transportProperties`` is
read directly by the NeoN C++ factory (``read_transport_viscosity``);
``constant/turbulenceProperties`` is loaded in ``create_fields`` to select the
turbulence model (pure-Python NeoN family, C++ fallback) — so no config class
is declared for them here.
"""

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import IOStrategy, OF


@IOStrategy(OF("system/controlDict"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict`` — the file-bound time/write config for this solver.

    Inherits the time-stepping + write schema from
    :class:`~neofoam.algorithms.solution_loop.config.TimeControlConfig`. There
    is no ``writeBackend`` knob: NeoN fields live outside the OpenFOAM
    objectRegistry, so the NeoN write hook is the only backend.

    Adaptive stepping (``adjustTimeStep``/``maxCo``/``maxDeltaT``) is **not**
    here: it co-owns ``controlDict`` through the opt-in time-step contribution
    models (``courant`` / ``maxDeltaT``), written only when those are active.
    """

    application: str = "neoPimpleFoam"
