# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Solver-core configs for incompressibleFluid.

Pydantic ``BaseConfig`` for ``system/controlDict``, loaded via
``@IOStrategy(OF(...))`` — it feeds validation of the time-stepping controls
before the solver opens any C++ runtime. ``constant/transportProperties`` is
**not** declared here: it is owned by the viscosity model
(:class:`neofoam.viscosity.config.TransportPropertiesConfig`), which the solver
binds as a core model family. The opt-in ``telemetry`` sub-dict is also a
solver-core config (not a model): activation is solver lifecycle, handled in
``run()`` before initialization.
"""

from neofoam.algorithms.solution_loop.config import TimeControlConfig
from neofoam.io import IOStrategy, OF, BaseConfig


@IOStrategy(OF("system/controlDict"))
class ControlDictConfig(TimeControlConfig):
    """``system/controlDict`` — the file-bound time/write config for this solver.

    Inherits the time-stepping + write schema (``startTime``/``endTime``/
    ``deltaT``/``writeControl``/``writeInterval``) from
    :class:`~neofoam.algorithms.solution_loop.config.TimeControlConfig`, so the framework
    stepper / loop / write control all consume one validated config. Adds only
    the solver-specific keys and binds the file via ``@IOStrategy``.

    Adaptive stepping (``adjustTimeStep``/``maxCo``/``maxDeltaT``) is **not** here:
    it co-owns ``controlDict`` through the opt-in time-step contribution models
    :class:`~neofoam.solver.incompressibleFluid.models.courant.CourantConfig` and
    :class:`~neofoam.solver.incompressibleFluid.models.max_delta_t.MaxDeltaTConfig`,
    written only when those models are active.
    """

    application: str = "pimpleFoam"
    # which FieldHook backend persists fields: "runtime" (registry write, the
    # default — captures sub-model/turbulence fields) or "perField" (writes the
    # write=True-flagged context fields individually).
    writeBackend: str = "runtime"


@IOStrategy(OF("system/controlDict", subdict="telemetry"))
class TelemetryDictConfig(BaseConfig):
    """The ``telemetry`` sub-dict of ``system/controlDict`` (all keys optional).

    Solver-owned opt-in for OpenTelemetry tracing: the dict being present (and
    ``enabled`` absent-or-true) activates tracing for the run. Activation
    happens in the solver's ``run()`` via ``maybe_configure_telemetry``
    *before* initialization, so init spans are captured too.
    """

    enabled: bool = True
    directory: str = "telemetry"
    summary: bool = True
