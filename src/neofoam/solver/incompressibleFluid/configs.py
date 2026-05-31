# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleFluid.

Pydantic ``BaseConfig`` for ``system/controlDict``, loaded via
``@IOStrategy(OF(...))`` — it feeds validation of the time-stepping controls
before the solver opens any C++ runtime. ``constant/transportProperties`` is
**not** declared here: it is owned by the viscosity model
(:class:`neofoam.viscosity.config.TransportPropertiesConfig`), which the solver
binds as a core model family.
"""

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import IOStrategy, OF


@IOStrategy(OF("system/controlDict"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict`` — the file-bound time/write config for this solver.

    Inherits the time-stepping + write schema (``startTime``/``endTime``/
    ``deltaT``/``adjustTimeStep``/``maxCo``/``maxDeltaT``/``writeControl``/
    ``writeInterval`` and the ``adjustTimeStep`` ⇒ ``maxCo`` invariant) from
    :class:`~neofoam.algorithms.solution_loop.config.TimeControlConfig`, so the framework
    stepper / loop / write control all consume one validated config. Adds only
    the solver-specific keys and binds the file via ``@IOStrategy``.
    """

    application: str = "pimpleFoam"
    # which FieldHook backend persists fields: "runtime" (registry write, the
    # default — captures sub-model/turbulence fields) or "perField" (writes the
    # write=True-flagged context fields individually).
    writeBackend: str = "runtime"
