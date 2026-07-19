# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Pydantic-based control condition classes for CFD algorithms.

This module provides:
- Generic iteration conditions (IterationCountCondition)
- Residual-based convergence checking (ResidualConvergenceCondition)
- Simple boolean flags (BooleanFlagCondition, SingleIterationCondition)
- Algorithm control bundles (PimpleControl, SimpleControl)

All classes are Pydantic BaseModels with automatic validation.
"""

from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class IterationCountCondition(BaseModel):
    """
    Generic iteration count condition.

    Executes a loop for a fixed number of iterations.
    Can be used for corrector loops, non-orthogonal loops, etc.

    Attributes:
        nIterations: Number of iterations (must be >= 1)
    """

    model_config = {"arbitrary_types_allowed": True}

    nIterations: int = Field(..., ge=1, description="Number of iterations to execute")

    _iteration_count: int = 0
    _linked_conditions: list[Any] = []

    def __call__(self, ctx: Any) -> bool:
        """
        Check if loop should continue.

        Returns:
            True if more iterations needed, False otherwise
        """
        if self._iteration_count < self.nIterations:
            self._iteration_count += 1

            # Reset any linked conditions on each outer iteration
            for linked in self._linked_conditions:
                linked.reset()

            return True
        return False

    def is_final(self) -> bool:
        """Check if this is the final iteration."""
        return self._iteration_count >= self.nIterations

    def reset(self) -> None:
        """Reset iteration counter."""
        self._iteration_count = 0

    def link_condition(self, condition: Any) -> None:
        """
        Link another condition to automatically reset when this one iterates.

        Args:
            condition: Condition to reset on each iteration of this condition
        """
        self._linked_conditions.append(condition)


class ResidualConvergenceCondition(BaseModel):
    """
    Residual-based convergence condition.

    Checks if all field residuals have dropped below specified tolerances.
    Returns False when converged (stop iteration), True to continue.

    Attributes:
        residualControl: Dictionary of field names to tolerance values
    """

    model_config = {"arbitrary_types_allowed": True}

    residualControl: dict[str, float] = Field(
        default_factory=dict,
        description="Field name to tolerance mapping (e.g., {'p': 1e-2, 'U': 1e-3})",
    )

    _converged: bool = False
    _get_residual: Optional[Callable[[Any, str], float]] = None

    def __call__(self, ctx: Any) -> bool:
        """
        Check convergence based on residuals.

        Returns:
            False if converged (stop iteration), True to continue
        """
        if not self.residualControl:
            # No residual control specified - continue iteration
            return True

        # Check all residuals
        all_converged = True
        for field_name, tolerance in self.residualControl.items():
            if self._get_residual is not None:
                residual = self._get_residual(ctx, field_name)
            else:
                # Default: extract from context (to be implemented with actual residual tracking)
                residual = self._extract_residual_from_context(ctx, field_name)

            if residual > tolerance:
                all_converged = False
                break

        self._converged = all_converged

        # Return False if converged (stop iteration), True to continue
        return not all_converged

    def converged(self) -> bool:
        """Check if converged."""
        return self._converged

    def reset(self) -> None:
        """Reset convergence flag."""
        self._converged = False

    def _extract_residual_from_context(self, ctx: Any, field_name: str) -> float:
        """
        Extract residual from context (placeholder for actual implementation).

        Args:
            ctx: Execution context
            field_name: Name of field

        Returns:
            Residual value
        """
        # This will be implemented when integrating with actual residual tracking
        # For now, return a large value to continue iteration
        return 1.0


class SingleIterationCondition(BaseModel):
    """
    Single iteration condition (executes once per reset).

    Returns True once, then False until reset.
    Used for PIMPLE outer loop (once per time step).
    """

    _executed: bool = False

    def __call__(self, ctx: Any) -> bool:
        """
        Execute once per reset.

        Returns:
            True on first call, False on subsequent calls
        """
        if not self._executed:
            self._executed = True
            return True
        return False

    def reset(self) -> None:
        """Reset for next execution."""
        self._executed = False


class BooleanFlagCondition(BaseModel):
    """
    Simple boolean flag condition.

    Returns the value of the enabled flag.
    Used for momentumPredictor, turbCorr, etc.

    Attributes:
        enabled: Whether condition is enabled
    """

    enabled: bool = Field(default=True, description="Whether condition is enabled")

    def __call__(self, ctx: Any) -> bool:
        """Return flag value."""
        return self.enabled


class PimpleControl(BaseModel):
    """
    PIMPLE algorithm control bundle.

    Manages nested loop structure:
    - Outer loop: nOuterCorrectors iterations (IterationCountCondition)
    - Corrector loop: nCorrectors iterations (IterationCountCondition)
    - Non-orthogonal loop: nNonOrthogonalCorrectors + 1 iterations

    Attributes:
        nCorrectors: Number of PIMPLE corrector iterations (>= 1)
        nNonOrthogonalCorrectors: Number of non-orthogonal corrections (>= 0)
        momentumPredictor_enabled: Whether to run momentum predictor
        turbCorr_enabled: Whether to apply turbulence correction
    """

    model_config = {"arbitrary_types_allowed": True}

    nCorrectors: int = Field(ge=2, description="Number of PIMPLE corrector iterations")
    nOuterCorrectors: int = Field(
        default=1, ge=1, description="Number of PIMPLE outer corrector iterations"
    )
    nNonOrthogonalCorrectors: int = Field(
        default=0, ge=0, description="Number of non-orthogonal corrections"
    )
    momentumPredictor_enabled: bool = Field(
        description="Enable momentum predictor", alias="momentumPredictor"
    )
    turbCorr_enabled: bool = Field(
        default=False, description="Enable turbulence correction", alias="turbCorr"
    )

    _loop: Optional[IterationCountCondition] = None
    _corrector: Optional[IterationCountCondition] = None
    _non_ortho: Optional[IterationCountCondition] = None
    _momentum_predictor: Optional[BooleanFlagCondition] = None
    _turb_corr: Optional[BooleanFlagCondition] = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize nested conditions after model validation."""
        # Outer loop: nOuterCorrectors per time step
        self._loop = IterationCountCondition(nIterations=self.nOuterCorrectors)

        # Corrector loop
        self._corrector = IterationCountCondition(nIterations=self.nCorrectors)

        # Non-orthogonal loop (nNonOrthogonalCorrectors + 1)
        self._non_ortho = IterationCountCondition(
            nIterations=self.nNonOrthogonalCorrectors + 1
        )

        # Link non-ortho loop to corrector (reset non-ortho on each corrector iteration)
        self._corrector.link_condition(self._non_ortho)

        # Flags
        self._momentum_predictor = BooleanFlagCondition(
            enabled=self.momentumPredictor_enabled
        )
        self._turb_corr = BooleanFlagCondition(enabled=self.turbCorr_enabled)

    def loop(self, ctx: Any = None) -> bool:
        """
        PIMPLE outer loop.

        Returns:
            True while outer iterations remain, False otherwise.
            Automatically resets all counters when the loop finishes
            so the control is ready for the next time step.
        """
        assert self._loop is not None
        result = self._loop(ctx)
        if not result:
            self.reset()
        return result

    def correct(self, ctx: Any = None) -> bool:
        """
        PIMPLE corrector loop.

        Returns:
            True while corrector iterations remain, False otherwise
        """
        assert self._corrector is not None
        return self._corrector(ctx)

    def correctNonOrthogonal(self, ctx: Any = None) -> bool:
        """
        Non-orthogonal correction loop.

        Returns:
            True while non-orthogonal corrections remain, False otherwise
        """
        assert self._non_ortho is not None
        return self._non_ortho(ctx)

    def finalIter(self) -> bool:
        """Check if this is the final outer (PIMPLE) iteration."""
        assert self._loop is not None
        return self._loop.is_final()

    def finalInnerIter(self) -> bool:
        """Check if this is the final corrector iteration."""
        assert self._corrector is not None
        return self._corrector.is_final()

    def finalNonOrthogonalIter(self) -> bool:
        """Check if this is the final non-orthogonal iteration."""
        assert self._non_ortho is not None
        return self._non_ortho.is_final()

    def momentumPredictor(self) -> bool:
        """Check if momentum predictor is enabled."""
        assert self._momentum_predictor is not None
        return self._momentum_predictor(None)

    def turbCorr(self) -> bool:
        """Check if turbulence correction is enabled."""
        assert self._turb_corr is not None
        return self._turb_corr(None)

    def reset(self) -> None:
        """Reset all conditions for next time step."""
        assert self._loop is not None
        assert self._corrector is not None
        assert self._non_ortho is not None
        self._loop.reset()
        self._corrector.reset()
        self._non_ortho.reset()


class SimpleControl(BaseModel):
    """
    SIMPLE algorithm control bundle.

        Manages loop structure:
        - Main loop: one pass per outer solver iteration by default
            (optionally convergence-driven when useResidualConvergence=True)
    - Non-orthogonal loop: nNonOrthogonalCorrectors + 1 iterations

    Attributes:
        nNonOrthogonalCorrectors: Number of non-orthogonal corrections (>= 0)
        residualControl: Dictionary of field tolerances for convergence
    """

    model_config = {"arbitrary_types_allowed": True}

    nNonOrthogonalCorrectors: int = Field(
        default=0, ge=0, description="Number of non-orthogonal corrections"
    )
    momentumPredictor_enabled: bool = Field(
        default=True, description="Enable momentum predictor", alias="momentumPredictor"
    )
    consistent_enabled: bool = Field(
        default=False, description="Enable SIMPLEC consistent mode", alias="consistent"
    )
    useResidualConvergence: bool = Field(
        default=False,
        description=(
            "Enable convergence-driven SIMPLE loop behavior "
            "(legacy mode with residualControl + optional runtime guard)"
        ),
    )
    residualControl: dict[str, float] = Field(
        default_factory=dict,
        description="Field tolerance mapping for convergence checking",
    )

    _residual_check: Optional[ResidualConvergenceCondition] = None
    _inner_loop_open: bool = True
    _non_ortho: Optional[IterationCountCondition] = None
    _momentum_predictor: Optional[BooleanFlagCondition] = None
    _consistent: Optional[BooleanFlagCondition] = None
    _iteration_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        """Initialize SIMPLE control conditions after model validation."""
        self._residual_check = ResidualConvergenceCondition(
            residualControl=self.residualControl
        )
        self._non_ortho = IterationCountCondition(
            nIterations=self.nNonOrthogonalCorrectors + 1
        )
        self._momentum_predictor = BooleanFlagCondition(
            enabled=self.momentumPredictor_enabled
        )
        self._consistent = BooleanFlagCondition(enabled=self.consistent_enabled)

    def loop(self, ctx: Any = None) -> bool:
        """
        SIMPLE loop gate.

        Default behavior (useResidualConvergence=False):
        - Returns True once per reset, then False.

        Legacy behavior (useResidualConvergence=True):
        - Continues iteration until converged or runtime loop stops.

        Returns:
            True to continue SIMPLE pass, False to stop
        """
        assert self._residual_check is not None

        if not self.useResidualConvergence:
            if self._inner_loop_open:
                self._inner_loop_open = False
                return True
            self._inner_loop_open = True
            # Closing the pass re-arms the non-orthogonal corrector for the
            # next outer iteration (mirrors PimpleControl's auto-reset —
            # without it only the first pass ever runs a pressure solve).
            assert self._non_ortho is not None
            self._non_ortho.reset()
            return False

        self._iteration_count += 1

        # Check convergence
        if not self._residual_check(ctx):
            # Converged (residual check returns False when converged)
            return False

        # Check runtime limit only if ctx is provided
        if ctx is not None and hasattr(ctx, "runTime") and hasattr(ctx.runTime, "loop"):
            if not ctx.runTime.loop():
                return False

        return True

    def correctNonOrthogonal(self, ctx: Any = None) -> bool:
        """
        Non-orthogonal correction loop.

        Returns:
            True while non-orthogonal corrections remain, False otherwise
        """
        assert self._non_ortho is not None
        return self._non_ortho(ctx)

    def finalNonOrthogonalIter(self) -> bool:
        """Check if this is the final non-orthogonal iteration."""
        assert self._non_ortho is not None
        return self._non_ortho.is_final()

    def momentumPredictor(self) -> bool:
        """Check if momentum predictor is enabled."""
        assert self._momentum_predictor is not None
        return self._momentum_predictor(None)

    def consistent(self) -> bool:
        """Check if SIMPLEC consistent mode is enabled."""
        assert self._consistent is not None
        return self._consistent(None)

    def converged(self) -> bool:
        """Check if algorithm has converged."""
        assert self._residual_check is not None
        return self._residual_check.converged()

    def get_residual_condition(self) -> ResidualConvergenceCondition:
        """
        Get residual convergence condition for advanced composition.

        Allows boolean composition with framework's Condition class:

        Example:
            simple = SimpleControl(mesh, residualControl={'p': 1e-2})
            residual_cond = Condition(simple.get_residual_condition(), "Residuals")
            custom_cond = ~residual_cond & other_condition

        Returns:
            ResidualConvergenceCondition instance
        """
        assert self._residual_check is not None
        return self._residual_check

    def reset(self) -> None:
        """Reset convergence tracking."""
        assert self._residual_check is not None
        assert self._non_ortho is not None
        self._residual_check.reset()
        self._inner_loop_open = True
        self._non_ortho.reset()
        self._iteration_count = 0


class SolutionControl(BaseModel):
    """Outer solution-loop control — *separate* from the algorithm controls.

    Owns the outer-loop predicate (advance time, end on convergence); the
    corrector-loop structure stays in :class:`PimpleControl` / :class:`SimpleControl`.
    Mirrors ``Foam::solutionControl`` + ``simpleControl::loop()``.

    ``residualControl`` empty  ⇒ transient: never ends early, runs to ``endTime``.
    ``residualControl`` set    ⇒ steady: ends the run once every field's residual
    drops below its tolerance (``endTime`` is then the iteration cap).
    """

    model_config = {"arbitrary_types_allowed": True}

    residualControl: dict[str, float] = Field(
        default_factory=dict,
        description="Field tolerance mapping for outer convergence (empty = transient)",
    )

    _residual_check: Optional[ResidualConvergenceCondition] = None
    _residuals: dict[str, float] = {}

    def model_post_init(self, __context: Any) -> None:
        self._residuals = {}
        self._residual_check = ResidualConvergenceCondition(
            residualControl=self.residualControl
        )
        # feed the residual check from the residuals the solver publishes
        # (replaces the placeholder that always returned 1.0 = never converged)
        self._residual_check._get_residual = lambda _ctx, field: self._residuals.get(
            field, 1.0
        )

    def store_residual(self, field: str, initial_residual: float) -> None:
        """Publish the initial residual of ``field`` for convergence checking."""
        self._residuals[field] = float(initial_residual)

    def converged(self) -> bool:
        """True once every ``residualControl`` field is below tolerance."""
        assert self._residual_check is not None
        self._residual_check(None)  # recompute from stored residuals
        return self._residual_check.converged()

    def run(self, loop: Any) -> bool:
        """Outer-loop predicate: govern advancement only.

        End the run on convergence (stop advancing), otherwise report whether
        there are more steps. It never writes — persisting fields is the
        ``WriteControl``'s responsibility — so ``stop()`` here means "end the
        run", not OpenFOAM's ``writeAndEnd``.

        ``SolutionLoop.running()`` calls this once per step, passing the loop
        itself (which exposes ``run()``/``stop()`` over its ``LoopState``):
        ``return self._control.run(self)``.
        """
        if self.converged():
            loop.stop()  # end the run (no write)
        return bool(loop.run())
