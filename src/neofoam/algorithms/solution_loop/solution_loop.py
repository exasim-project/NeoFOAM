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
  write-time decision all live here) and delegates the outer-loop predicate to a
  ``control`` (:class:`~neofoam.algorithms.solution_loop.control.SolutionControl`).
  Stability limits are folded per step from the model-owned ``timeStepConstraint``
  / ``maxTimeStep`` / ``initialTimeStepConstraint`` interfaces (the active
  ``courant``/``maxDeltaT`` contributors) — there is no ``set_courant`` drive.
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
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from neofoam.algorithms.solution_loop.config import (
    _ADJUSTABLE,
    _RUN_TIME,
    _TIME_STEP,
    _WRITE_CONTROL_ALIASES,
    LABEL_MAX,
    SMALL,
    TimeControlConfig,
    _round_half_away,
)
from neofoam.algorithms.solution_loop.control import SolutionControl
from neofoam.algorithms.solution_loop.interfaces import (
    VGREAT,
    initialTimeStepConstraint,
    loopCondition,
    maxTimeStep,
    timeStepConstraint,
)

# Re-export the solutionLoop Model declared in interfaces.py (the leaf module);
# the explicit alias marks it as a public re-export for the solver-side module
# that imports it from here (mypy strict no_implicit_reexport).
from neofoam.algorithms.solution_loop.interfaces import solutionLoop as solutionLoop
from neofoam.algorithms.solution_loop.loop_state import LoopState
from neofoam.algorithms.solution_loop.time_integration import (
    TimeIntegration,
    TransientIntegration,
)
from neofoam.framework.context import Context
from neofoam.framework.initialization import InitStep, lazy, model


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
            loop.constrain_delta_t(timeStepConstraint())  # folded model limits
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
        conditions: Optional[list[Callable[["SolutionLoop"], bool]]] = None,
        growth_cap: float = 1.2,
        backend: Optional[LoopBackend] = None,
    ) -> None:
        self._state = state
        self._integration: TimeIntegration = (
            integration if integration is not None else TransientIntegration()
        )
        self._control: LoopControl = control if control is not None else SolutionControl()
        self._conditions: list[Callable[["SolutionLoop"], bool]] = list(conditions or [])
        # latest interface folds, written by the set_time_step operation: next_dt
        # is the folded timeStepConstraint limit and keep_running the loopCondition
        # fold, which the outer-loop predicate ANDs with running() to stop the run.
        self.next_dt: float = VGREAT
        self.keep_running: bool = True
        self._growth_cap = growth_cap
        self._backend: LoopBackend = backend if backend is not None else NullLoopBackend()

    # -- injection --------------------------------------------------------
    def set_backend(self, backend: LoopBackend) -> None:
        """Inject the backend after construction (e.g. a solver wiring its
        ``Foam::Time`` mirror). Standalone use keeps :class:`NullLoopBackend`."""
        self._backend = backend
        self._backend.update(self._state)

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

    def set_initial_delta_t(self, limit: float, ceiling: float = VGREAT) -> None:
        """``setInitialDeltaT.H`` — the undamped first-step reduction.

        Runs once, before the first ``constrain_delta_t``, and only on a
        non-quiescent flow (the contributors express that gate by offering no
        ``initialTimeStepConstraint`` opinion when ``Co <= SMALL``, exactly as
        ``setInitialDeltaT.H`` skips its body). It can only *lower* the step —
        ``min(limit, deltaT, maxDeltaT)`` — but it does go through
        :meth:`set_delta_t`, so the write-time snapping runs before the damped
        pass sees the step.
        """
        if limit >= VGREAT:
            return
        self.set_delta_t(min(limit, min(self._state.delta_t, ceiling)))

    def constrain_delta_t(self, limit: float, ceiling: float = VGREAT) -> None:
        """Set the next ``deltaT`` from the folded loop-limit interfaces.

        A faithful ``setDeltaT.H``: both limits ``VGREAT`` = no active opinion
        (``adjustTimeStep no``), and the step is left alone — OpenFOAM never
        reaches ``Time::setDeltaT`` then, so the write-time snapping must not run
        either. Otherwise the Courant-style *limit* is approached the way
        ``setDeltaT.H`` approaches it — shrinking takes it at once, growing is
        damped through ``min(fact, 1 + 0.1*fact, growth_cap)`` with
        ``fact = limit / current`` — and the ``maxTimeStep`` *ceiling* is clipped
        onto the damped result afterwards, never pushed through the damping (the
        two do not commute). The result is snapped onto the next write time.
        """
        current = self._state.delta_t
        if limit >= VGREAT and ceiling >= VGREAT:
            return
        fact = limit / current
        damped = min(fact, 1.0 + 0.1 * fact, self._growth_cap) * current
        self.set_delta_t(min(damped, ceiling))

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
            write_index = int(((s.value - s.start_time) + 0.5 * s.delta_t) / s.write_interval)
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
    def conditions(self) -> list[Callable[["SolutionLoop"], bool]]:
        return list(self._conditions)

    def all_conditions_hold(self) -> bool:
        """Fold the constructor-injected conditions with ``all`` (empty → True)."""
        return all(c(self) for c in self._conditions)


# =========================================================================
# The core Model wrapping the engine
# =========================================================================
#
# ``solutionLoop`` is declared in ``interfaces.py`` (the leaf module that also owns
# its two model-owned interfaces); the decorators below mutate that imported object.


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
        write_control=_WRITE_CONTROL_ALIASES.get(config.writeControl, config.writeControl),
        write_interval=config.writeInterval,
    )


def make_solution_loop(
    config: TimeControlConfig,
    state: LoopState,
    *,
    integration: Optional[TimeIntegration] = None,
    control: Optional[Any] = None,
) -> SolutionLoop:
    """Wrap the state in the loop engine.

    Stability limits are no longer seeded here: they are folded per step from the
    model-owned ``timeStepConstraint`` interface (the active ``courant``/``maxDeltaT``
    contributors). ``residualControl`` is an fvSolution concern, so steady
    convergence is wired by passing a configured ``control``.
    """
    if integration is None:
        integration = TransientIntegration()
    return SolutionLoop(
        state=state,
        integration=integration,
        control=control if control is not None else SolutionControl(),
    )


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
    never special-cases steady vs unsteady. The run also ends when the folded
    ``loopCondition`` (recorded as ``keep_running`` by ``set_time_step``,
    default ``True``) votes to stop.
    """

    def __call__(self, ctx: Context) -> bool:
        loop = _engine(ctx)
        return bool(loop.running() and loop.keep_running)


# -- operations: the loop body --------------------------------------------
@solutionLoop.operation()
def set_time_step(
    self: Any,
    ctx: Context,
    constraints: timeStepConstraint,  # type: ignore[valid-type]
    ceilings: maxTimeStep,  # type: ignore[valid-type]
    initial_constraints: initialTimeStepConstraint,  # type: ignore[valid-type]
    conditions: loopCondition,  # type: ignore[valid-type]
) -> None:
    """Set the next ``deltaT`` from the folded limit interfaces and record the
    loop continue-flag from ``loopCondition``.

    Publishes the loop's current step as ``ctx.fields["deltaT"]`` so a contribution
    can inject ``deltaT`` without reading the Context, then **calls** each injected
    interface with the **live** Context so the active contributing models resolve
    their fields/config at this step. No active constraint -> ``min`` default
    ``VGREAT`` -> the step is unchanged (fixed step).

    The first step runs the ``setInitialDeltaT.H`` pass before the ``setDeltaT.H``
    one, and re-publishes ``deltaT`` in between, mirroring the ``CourantNo.H`` that
    pimpleFoam/interFoam re-run at the head of the loop: the Courant number the
    damped pass sees is the one belonging to the step the initial pass left behind.
    """
    loop = _engine(ctx)
    ctx.fields["deltaT"] = loop.current_delta_t()
    loop.keep_running = conditions(ctx)  # type: ignore[misc]
    ceiling = ceilings(ctx)  # type: ignore[misc]
    if loop.state.index == 0:
        loop.set_initial_delta_t(initial_constraints(ctx), ceiling)  # type: ignore[misc]
        ctx.fields["deltaT"] = loop.current_delta_t()
    limit = constraints(ctx)  # type: ignore[misc]
    loop.next_dt = limit
    loop.constrain_delta_t(limit, ceiling)


@solutionLoop.operation()
def increment_time(self: Any, ctx: Context) -> None:
    """Announce the current step and advance the loop one step."""
    loop = _engine(ctx)
    logger = ctx.models.get("loop_logger")
    (logger if logger is not None else print)(f"Time = {loop.timeName()}")
    loop.advance()
