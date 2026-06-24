# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the model operations take
# ``ctx: Context`` and the dependency_resolver matches ``param.annotation is
# Context``; stringized annotations would break that injection.

"""The main iteration loop — engine *and* its core Model, in one module.

Three collaborating pieces of one concern:

* :class:`~neofoam.algorithms.solution_loop.loop_state.LoopState` — the inspectable
  *data* (time, deltaT, index, write flag, …). Exposed as ``ctx.time``.
* :class:`SolutionLoop` — owns the *logic*. It advances the ``LoopState`` (the
  Foam::Time ``operator++`` arithmetic, ``adjustableRunTime`` snapping and
  write-time decision all live here), takes the ``min`` over injectable
  :class:`~neofoam.algorithms.constraints.time_step.DeltaTConstraint`s for the next
  ``deltaT``, and delegates the outer-loop predicate to a ``control``
  (:class:`~neofoam.algorithms.solution_loop.control.SolutionControl`). The Courant
  number is *pushed in* via :meth:`set_courant`.
* :class:`LoopBackend` — an optional backend (default :class:`NullLoopBackend`,
  a no-op) that mirrors the advanced ``LoopState`` onto a real backend clock. The
  solver's pybFoam ``FoamTime`` implements it to keep ``Foam::Time`` in step so
  field IO lands in the right time directory; standalone/NeoN runs keep the null
  backend (no coupling).

:data:`solutionLoop` is the framework **core Model** wrapping the engine: it
``@load``s :class:`~neofoam.algorithms.solution_loop.config.TimeControlConfig` from
``system/controlDict``, ``@build``s the ``LoopState`` (as ``ctx.time``) and the
engine, and exposes the loop body as ``@operation``s plus the predicate.
"""

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from neofoam.algorithms.constraints.time_step import (
    DeltaTConstraint,
    next_delta_t,
)
from neofoam.algorithms.solution_loop.config import (
    LABEL_MAX,
    SMALL,
    _ADJUSTABLE,
    _RUN_TIME,
    _TIME_STEP,
    _WRITE_CONTROL_ALIASES,
    TimeControlConfig,
    _round_half_away,
)
from neofoam.algorithms.solution_loop.control import SolutionControl
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.time_integration import (
    TimeIntegration,
    TransientIntegration,
)
from neofoam.framework.context import Context
from neofoam.framework.initialization import InitStep, lazy, model
from neofoam.framework.model import Model


@runtime_checkable
class LoopControl(Protocol):
    """The outer-loop predicate: advance, end the run on convergence."""

    def run(self, loop: Any) -> bool: ...


@runtime_checkable
class LoopBackend(Protocol):
    """Optional backend that mirrors the advanced :class:`LoopState`.

    The solver's pybFoam ``FoamTime`` implements this to set the OpenFOAM
    ``Foam::Time`` internal state. ``update`` is called after every state change
    (deltaT set, step advanced); the implementation reconciles its own clock.
    """

    def update(self, state: LoopState) -> None: ...


class NullLoopBackend:
    """Default backend for standalone / pure-Python use: mirrors nothing."""

    def update(self, state: LoopState) -> None:
        return None


class SolutionLoop:
    """The pure-Python main iteration loop engine (owns the advancement logic).

    Canonical drive (steady and unsteady alike)::

        while loop.running():
            loop.publish("courant", measured_Co)   # if a model measures it
            loop.adjust_delta_t()
            loop.advance()
            ...solve...
            # the fieldWriter Model persists fields — not the loop's job
    """

    def __init__(
        self,
        *,
        state: LoopState,
        integration: Optional[TimeIntegration] = None,
        control: Optional[LoopControl] = None,
        constraints: Optional[list[DeltaTConstraint]] = None,
        growth_cap: float = 1.2,
        backend: Optional[LoopBackend] = None,
    ) -> None:
        self._state = state
        self._integration: TimeIntegration = (
            integration if integration is not None else TransientIntegration()
        )
        self._control: LoopControl = (
            control if control is not None else SolutionControl()
        )
        self._constraints: list[DeltaTConstraint] = list(constraints or [])
        self._growth_cap = growth_cap
        self._measured: dict[str, float] = {}
        self._backend: LoopBackend = (
            backend if backend is not None else NullLoopBackend()
        )

    # -- injection --------------------------------------------------------
    def add_constraint(self, constraint: DeltaTConstraint) -> "SolutionLoop":
        """Inject a stability criterion (any model's deltaT limit)."""
        self._constraints.append(constraint)
        return self

    def set_backend(self, backend: LoopBackend) -> None:
        """Inject the backend after construction (e.g. a solver wiring its
        ``Foam::Time`` mirror). Standalone use keeps :class:`NullLoopBackend`."""
        self._backend = backend
        self._backend.update(self._state)

    def publish(self, name: str, value: float) -> None:
        """Publish a measured quantity (e.g. ``"courant"``) for constraints to read.

        A time-step model measures a quantity on the live fields each step and
        publishes it under a name; the :class:`DeltaTConstraint` that declares
        ``measures = name`` reads it back via :meth:`measured`. This is the
        open seam that lets a new stability rule carry its own measurement
        without the engine knowing the quantity.
        """
        self._measured[name] = float(value)

    # -- constraint context: what a DeltaTConstraint reads ----------------
    def measured(self, name: str, default: float = 0.0) -> float:
        """The last value published under ``name`` (``default`` if never set)."""
        return self._measured.get(name, default)

    def current_delta_t(self) -> float:
        return self._state.delta_t

    # -- run / end test on the state (the SolutionControl drives these) ---
    def run(self) -> bool:
        s = self._state
        return s.value < (s.end_time - 0.5 * s.delta_t)

    def end(self) -> bool:
        s = self._state
        return s.value > (s.end_time + 0.5 * s.delta_t)

    def stop(self) -> None:
        """End the run now — advancement only, no write (the WriteControl owns
        whether the final step is written)."""
        self._state.end_time = self._state.value

    def timeName(self) -> str:
        return self._integration.step_name(
            self._state.value, self._state.index, self._state.precision
        )

    # -- the steps of one iteration ---------------------------------------
    def running(self) -> bool:
        """Outer-loop predicate (delegated to the SolutionControl)."""
        return bool(self._control.run(self))

    def set_delta_t(self, dt: float, adjust: bool = True) -> None:
        """Set the next ``deltaT`` (with ``adjustableRunTime`` snapping), then
        mirror onto the backend."""
        self._state.delta_t = dt
        if adjust:
            self._snap_delta_t()
        self._backend.update(self._state)

    def _snap_delta_t(self) -> None:
        """``Foam::Time::adjustDeltaT`` — snap to land on the next write time."""
        s = self._state
        if s.write_control != _ADJUSTABLE:
            return
        time_to_next_write = max(
            0.0,
            (s.write_time_index + 1) * s.write_interval - (s.value - s.start_time),
        )
        n_steps = time_to_next_write / s.delta_t
        if n_steps < LABEL_MAX:
            n_steps_to_next = max(1, _round_half_away(n_steps))
            new_delta_t = time_to_next_write / n_steps_to_next
            if new_delta_t >= s.delta_t:
                s.delta_t = min(new_delta_t, 2.0 * s.delta_t)
            else:
                s.delta_t = max(new_delta_t, 0.2 * s.delta_t)

    def adjust_delta_t(self) -> None:
        """Set the next ``deltaT`` = min over every injected constraint."""
        dt = next_delta_t(
            self._constraints,
            self,
            current_dt=self._state.delta_t,
            growth_cap=self._growth_cap,
        )
        self.set_delta_t(dt)

    def advance(self) -> None:
        """``Foam::Time::operator++`` — advance, roll the old time, set writeTime,
        then mirror onto the backend."""
        s = self._state
        s.delta_t0 = s.delta_t_save
        s.delta_t_save = s.delta_t

        s.value = s.value + s.delta_t
        s.index += 1

        if abs(s.value) < 10 * SMALL * s.delta_t:
            s.value = 0.0

        s.write_time = False
        if s.write_control == _TIME_STEP:
            s.write_time = (s.index % int(s.write_interval)) == 0
        elif s.write_control in (_RUN_TIME, _ADJUSTABLE):
            write_index = int(
                ((s.value - s.start_time) + 0.5 * s.delta_t) / s.write_interval
            )
            if write_index > s.write_time_index:
                s.write_time = True
                s.write_time_index = write_index

        self._backend.update(s)

    # -- accessors --------------------------------------------------------
    @property
    def state(self) -> LoopState:
        """The live :class:`LoopState` (same object as ``ctx.time``)."""
        return self._state

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


def make_loop_state(
    config: TimeControlConfig, *, integration: Optional[TimeIntegration] = None
) -> LoopState:
    """The initial :class:`LoopState` from a validated controlDict config."""
    if integration is None:
        integration = TransientIntegration()
    return LoopState(
        value=config.startTime,
        delta_t=integration.initial_delta_t(config.deltaT),
        end_time=config.endTime,
        start_time=config.startTime,
        write_control=_WRITE_CONTROL_ALIASES.get(
            config.writeControl, config.writeControl
        ),
        write_interval=config.writeInterval,
    )


def make_solution_loop(
    config: TimeControlConfig,
    state: LoopState,
    *,
    integration: Optional[TimeIntegration] = None,
    control: Optional[Any] = None,
) -> SolutionLoop:
    """Wrap the state in a **bare** loop engine — no stability constraints.

    Time-step constraints are no longer baked into the core loop: each is owned
    by an opt-in time-step model that installs its
    :class:`~neofoam.algorithms.constraints.time_step.DeltaTConstraint` into the
    built engine (see :func:`install_constraints_step`) and publishes any measured
    quantity it needs. A run with no such model selected keeps a fixed step; the
    engine's min-aggregation leaves an empty constraint list unchanged.
    ``residualControl`` is an fvSolution concern, so steady convergence is wired
    by passing a configured ``control``.
    """
    if integration is None:
        integration = TransientIntegration()
    return SolutionLoop(
        state=state,
        integration=integration,
        control=control if control is not None else SolutionControl(),
    )


def install_constraints_step(
    name: str,
    build_constraints: Any,
    *,
    depends_on: tuple[str, ...] = (),
) -> InitStep:
    """An :class:`InitStep` that adds a model's deltaT constraints to the engine.

    ``build_constraints(ctx)`` returns the
    :class:`~neofoam.algorithms.constraints.time_step.DeltaTConstraint`\\ s to
    install (it may read the build ``ctx``). The step depends on the built
    ``models.solution_loop`` so it runs after the engine exists, mirroring how
    ``loop_backend_steps`` injects the backend. This is the one-liner a time-step
    model's ``@build`` uses so a new stability rule never edits the core loop.
    """

    def _install(ctx: dict[str, Any]) -> SolutionLoop:
        loop: SolutionLoop = ctx["models.solution_loop"]
        for constraint in build_constraints(ctx):
            loop.add_constraint(constraint)
        return loop

    return model(name, _install, depends_on=["models.solution_loop", *depends_on])


# -- build: LoopState (ctx.time) + engine are this model's runtime state ---
@solutionLoop.build
def build(config: TimeControlConfig) -> list[InitStep]:
    def create_state(ctx: dict[str, Any]) -> LoopState:
        return make_loop_state(config)

    def create_engine(ctx: dict[str, Any]) -> SolutionLoop:
        return make_solution_loop(config, ctx["time"])

    return [
        lazy("time", create_state),
        model("solution_loop", create_engine, depends_on=["time"]),
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
    present, every registered ``measurement_provider.<name>`` model is consulted
    and its value published under ``<name>`` for the constraints to read — so a
    new time-step rule supplies its own measurement without the loop knowing it.
    """
    loop = _engine(ctx)
    if loop.constraints:
        for key, provider in ctx.models.items():
            if key.startswith("measurement_provider."):
                loop.publish(key[len("measurement_provider.") :], float(provider(ctx)))
    loop.adjust_delta_t()


@solutionLoop.operation()
def increment_time(self: Any, ctx: Context) -> None:
    """Announce the current step and advance the loop one step."""
    loop = _engine(ctx)
    logger = ctx.models.get("loop_logger")
    (logger if logger is not None else print)(f"Time = {loop.timeName()}")
    loop.advance()
