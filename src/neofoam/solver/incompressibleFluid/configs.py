# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleFluid.

Pydantic ``BaseConfig`` classes for ``system/controlDict`` and
``constant/transportProperties``, loaded via ``@IOStrategy(OF(...))``.
These feed validation of time-stepping and fluid properties before the
solver opens any C++ runtime.
"""

from pydantic import Field

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import BaseConfig, IOStrategy, OF


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


@IOStrategy(OF("constant/transportProperties"))
class TransportPropertiesConfig(BaseConfig):
    """Fluid transport properties from ``constant/transportProperties``."""

    transportModel: str = "Newtonian"
    nu: float = Field(gt=0)
