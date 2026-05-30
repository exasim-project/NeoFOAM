# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the model operations take
# ``ctx: Context`` and the dependency_resolver matches ``param.annotation is
# Context``; stringized annotations would break that injection.

"""The main iteration loop — engine *and* its core Model, in one module.

Two collaborating pieces of one concern, kept together:

* :class:`SolutionLoop` — the pure-Python loop **engine**. It advances the
  step (physical time *or* iteration via the stepper), takes the ``min`` over
  injectable :class:`~neofoam.algorithms.time_step.DeltaTConstraint`s for the next
  ``deltaT``, and delegates the outer-loop predicate to a ``control``
  (:class:`~neofoam.algorithms.control.SolutionControl`). It owns no policy and no
  backend; the Courant number is *pushed in* via :meth:`set_courant`.
* :data:`solutionLoop` — the framework **core Model** wrapping that engine: it
  ``@load``s :class:`~neofoam.algorithms.foam_time.TimeControlConfig` from
  ``system/controlDict``, ``@build``s the stepper + engine, and exposes the loop
  body as ``@operation``s (``set_time_step`` / ``increment_time``) plus the
  predicate.

The Model is **backend-agnostic** (imports no pybFoam/NeoN). Backend
touch-points are injected via ``ctx.models`` with no-op defaults: an optional
``courant_provider`` callable ``(Context) -> float`` (CFL on the live fields), an
optional ``loop_logger`` callable ``(str) -> None`` (defaults to :func:`print`),
and the stepper's :class:`~neofoam.algorithms.foam_time.StepSink`, injected
post-build via :meth:`~neofoam.algorithms.foam_time.FoamTime.set_sink`. A solver
registers the backend versions; persisting fields is the separate ``fieldWriter``
Model's concern.
"""

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from neofoam.algorithms.control import SolutionControl
from neofoam.algorithms.foam_time import FoamTime, TimeControlConfig
from neofoam.algorithms.time_step import (
    CourantConstraint,
    DeltaTConstraint,
    MaxDeltaTConstraint,
    next_delta_t,
)
from neofoam.algorithms.time_integration import TimeIntegration, TransientIntegration
from neofoam.framework.context import Context
from neofoam.framework.initialization import InitStep, model
from neofoam.framework.model import Model


@runtime_checkable
class LoopControl(Protocol):
    """The outer-loop predicate: advance, end the run on convergence."""

    def run(self, runtime: Any) -> bool: ...


class SolutionLoop:
    """The pure-Python main iteration loop engine.

    Canonical drive (steady and unsteady alike)::

        while loop.running():
            loop.set_courant(measured_Co)   # if a model measures it
            loop.adjust_delta_t()
            loop.advance()
            ...solve...
            # the fieldWriter Model persists fields — not the loop's job
    """

    def __init__(
        self,
        *,
        stepper: FoamTime,
        control: Optional[LoopControl] = None,
        constraints: Optional[list[DeltaTConstraint]] = None,
        growth_cap: float = 1.2,
    ) -> None:
        self._stepper = stepper
        self._control: LoopControl = (
            control if control is not None else SolutionControl()
        )
        self._constraints: list[DeltaTConstraint] = list(constraints or [])
        self._growth_cap = growth_cap
        self._courant = 0.0

    # -- injection --------------------------------------------------------
    def add_constraint(self, constraint: DeltaTConstraint) -> "SolutionLoop":
        """Inject a stability criterion (any model's deltaT limit)."""
        self._constraints.append(constraint)
        return self

    def set_courant(self, courant: float) -> None:
        """Publish the current Courant number (measured on the live fields).

        A flow model computes it from the backend fields and pushes it here; the
        injected :class:`CourantConstraint` then reads it via :meth:`max_courant`.
        """
        self._courant = float(courant)

    # -- constraint context: what a DeltaTConstraint reads ----------------
    def max_courant(self) -> float:
        return self._courant

    def current_delta_t(self) -> float:
        return self._stepper.deltaTValue()

    # -- the steps of one iteration ---------------------------------------
    def running(self) -> bool:
        """Outer-loop predicate (delegated to the SolutionControl)."""
        return bool(self._control.run(self._stepper))

    def adjust_delta_t(self) -> None:
        """Set the next ``deltaT`` = min over every injected constraint."""
        dt = next_delta_t(
            self._constraints,
            self,
            current_dt=self._stepper.deltaTValue(),
            growth_cap=self._growth_cap,
        )
        self._stepper.setDeltaT(dt)

    def advance(self) -> None:
        """Advance the stepper by one step (``++runTime``)."""
        self._stepper.increment()

    # -- accessors --------------------------------------------------------
    @property
    def stepper(self) -> FoamTime:
        """The stepper — read ``t``/``dt``/``dt0`` via ``value``/``deltaTValue``/``deltaT0Value``."""
        return self._stepper

    @property
    def control(self) -> LoopControl:
        return self._control

    @property
    def constraints(self) -> list[DeltaTConstraint]:
        return list(self._constraints)


# =========================================================================
# The core Model wrapping the engine
# =========================================================================

solutionLoop = Model("solutionLoop")


# -- load: the controlDict parameterises the loop -------------------------
@solutionLoop.load
def load(case_dir: Path, instance_id: str) -> TimeControlConfig:
    return TimeControlConfig.load(case_dir=case_dir)


def make_stepper(
    config: TimeControlConfig, *, integration: Optional[TimeIntegration] = None
) -> FoamTime:
    """The pure-Python stepper (NullStepSink by default; inject a backend sink
    afterwards via :meth:`FoamTime.set_sink`)."""
    return FoamTime.from_config(config, integration=integration)


def make_solution_loop(
    config: TimeControlConfig,
    stepper: FoamTime,
    *,
    integration: Optional[TimeIntegration] = None,
    control: Optional[Any] = None,
) -> SolutionLoop:
    """Wrap the stepper in the loop engine, seeding deltaT constraints from config.

    The single construction path (there is no engine ``from_config``): a
    transient adjustable run gets a :class:`CourantConstraint` (and a ``maxDeltaT``
    cap when set), min-aggregated by the engine; a steady run gets none (fixed
    pseudo-step). ``residualControl`` is an fvSolution concern, so steady
    convergence is wired by passing a configured ``control``; with none, the run
    is transient.
    """
    if integration is None:
        integration = TransientIntegration()
    loop = SolutionLoop(
        stepper=stepper,
        control=control if control is not None else SolutionControl(),
    )
    if isinstance(integration, TransientIntegration) and config.adjustTimeStep:
        if config.maxCo:
            loop.add_constraint(CourantConstraint(maxCo=float(config.maxCo)))
        if config.maxDeltaT:
            loop.add_constraint(MaxDeltaTConstraint(maxDeltaT=float(config.maxDeltaT)))
    return loop


# -- build: stepper + engine are this model's runtime state ---------------
@solutionLoop.build
def build(config: TimeControlConfig) -> list[InitStep]:
    def create_stepper(ctx: dict[str, Any]) -> FoamTime:
        return make_stepper(config)

    def create_engine(ctx: dict[str, Any]) -> SolutionLoop:
        return make_solution_loop(config, ctx["models.stepper"])

    return [
        model("stepper", create_stepper),
        model("solution_loop", create_engine, depends_on=["models.stepper"]),
    ]


def _engine(ctx: Context) -> SolutionLoop:
    """The loop engine built by :func:`build`, stored in the context."""
    return ctx.models["solution_loop"]  # type: ignore[no-any-return]


class SolutionLoopPredicate:
    """Outer-loop predicate — delegates to the ``solution_loop`` engine.

    The engine's :class:`SolutionControl` advances a transient run to
    ``endTime`` and ends a steady run on residual convergence, so the solver
    never special-cases steady vs unsteady.
    """

    def __call__(self, ctx: Context) -> bool:
        return bool(_engine(ctx).running())


# -- operations: the loop body --------------------------------------------
@solutionLoop.operation()
def set_time_step(self: Any, ctx: Context) -> None:
    """Set the next ``deltaT`` from the injected stability constraints.

    No steady/transient branch: a fixed-step run has no constraints, so the
    engine's min-aggregation leaves the step unchanged. When constraints are
    present and a ``courant_provider`` is injected, the measured Courant number
    is pushed into the engine for the CFL constraint to read.
    """
    loop = _engine(ctx)
    if loop.constraints:
        provider = ctx.models.get("courant_provider")
        if provider is not None:
            loop.set_courant(float(provider(ctx)))
    loop.adjust_delta_t()


@solutionLoop.operation()
def increment_time(self: Any, ctx: Context) -> None:
    """Announce the current step and advance the stepper one step."""
    loop = _engine(ctx)
    logger = ctx.models.get("loop_logger")
    (logger if logger is not None else print)(f"Time = {loop.stepper.timeName()}")
    loop.advance()
