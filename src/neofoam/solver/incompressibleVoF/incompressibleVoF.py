# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
incompressibleVoF solver - interFoam-style VoF solver using the NeoFOAM framework.

Mirrors the incompressibleFluid.py pattern with an extra alpha_advection step
before the momentum-continuity inner loop.
"""

from typing import Annotated, Optional, Protocol

import pybFoam as pyf
import pybFoam.vof as vof
from pybFoam import Info, Time
from pybFoam import (
    surfaceScalarField,
    volScalarField,
    volVectorField,
)

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInit
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    OperationCollection,
    StepBuilder,
)
from neofoam.framework.solver_factory import Solver

from .create_fields import create_init


class CorrectableModel(Protocol):
    def correct(self) -> None: ...


class TimeLoop:
    """Helper: checks whether the time loop should continue."""

    def __call__(self, ctx: Context) -> bool:
        runTime: Time = ctx.runTime
        return bool(runTime.run())


# Create Solver instance
incompressibleVoF = Solver("incompressibleVoF")


@incompressibleVoF.initializer
def initialize(init: Annotated[StagedInit, Depends(create_init)]) -> Context:
    """Initialize fields and models via staged init."""
    init.argv = incompressibleVoF.argv
    ctx = init.run()
    incompressibleVoF.core_models = init.core_models
    incompressibleVoF.optional_models = init.optional_models
    return ctx


@incompressibleVoF.execution_graph_step
def execution_graph(
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, OperationCollection]:
    """Build solver structure for VoF: alpha_advection -> momentum -> continuity."""
    _ = domain_name

    ops = incompressibleVoF.operations
    builder = StepBuilder()

    algorithm_model = incompressibleVoF.core_models[0]
    pimple_model = incompressibleVoF.core_models[1]
    alpha_ops = algorithm_model.operations_for(algorithm_model)
    pimple_ops = pimple_model.operations_for(pimple_model)

    time_loop_op = Operation(
        func=IterativeOp(TimeLoop()),
        operation_name="time_loop",
        operation_number=None,
    )

    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(ops["set_time_step"])
        time_builder.step(ops["increment_time"])

        with time_builder.loop(pimple_ops["inner_loop"]) as inner_builder:
            inner_builder.step(alpha_ops["alpha_advection"])
            inner_builder.step(pimple_ops["momentum"])
            inner_builder.step(pimple_ops["continuity"])
            inner_builder.step(ops["turbulence_correction"])

        time_builder.step(ops["write_output"])

    model_ops = OperationCollection()
    for model in incompressibleVoF.optional_models:
        model_ops.add(model.operations)

    return builder, model_ops


def run(argv: Optional[list[str]] = None) -> Context:
    """Run the complete incompressibleVoF simulation.

    Args:
        argv: Command-line arguments (passed through to OpenFOAM)

    Returns:
        Final context after solving
    """
    incompressibleVoF.argv = argv or []
    ctx = incompressibleVoF.initialize()

    Info("Starting time loop")

    builder, model_ops = incompressibleVoF.execution_graph()
    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)

    resolved.operations.run(ctx)

    Info("End")
    return ctx


# ---------------------------------------------------------------------------
# Solver operations
# ---------------------------------------------------------------------------


@incompressibleVoF.operation()
def set_time_step(
    self,
    ctx: Context,
    phi: surfaceScalarField,
    alpha1: volScalarField,
) -> None:
    """Adjust time step using both flow CFL and alpha CFL (adaptive dt)."""
    runTime: Time = ctx.runTime

    # Read controlDict using pybFoam dictionary proxy
    ctrl_dict = runTime.controlDict()

    def _dict_bool(key: str, default: bool) -> bool:
        if ctrl_dict.found(key):
            try:
                return ctrl_dict.getOrDefault[bool](key, default)
            except Exception:
                pass
        return default

    def _dict_float(key: str, default: float) -> float:
        if ctrl_dict.found(key):
            try:
                return float(ctrl_dict.getOrDefault[float](key, default))
            except Exception:
                pass
        return default

    if not _dict_bool("adjustTimeStep", False):
        return

    max_co = _dict_float("maxCo", 1.0)
    max_alpha_co = _dict_float("maxAlphaCo", 1.0)
    max_delta_t = _dict_float("maxDeltaT", 1.0)

    # Flow Courant number (max, mean)
    maxCoNum, meanCoNum = pyf.computeCFLNumber(phi)
    # Alpha (interface) Courant number (max, mean)
    alphaCoNum, meanAlphaCo = vof.computeAlphaCourantNumber(phi, alpha1)

    Info(f"Courant Number mean: {meanCoNum:.4g}  max: {maxCoNum:.4g}")
    Info(f"Interface Courant Number mean: {meanAlphaCo:.4g}  max: {alphaCoNum:.4g}")

    # Mirror interFoam setDeltaT.H: limit by both Co constraints, allow ramped
    # increase (at most +20% per step) when Co is well below the limit.
    current_dt = runTime.deltaTValue()
    max_delta_t_fact = min(
        max_co / (maxCoNum + 1e-15),
        max_alpha_co / (alphaCoNum + 1e-15),
    )
    delta_t_fact = min(min(max_delta_t_fact, 1.0 + 0.1 * max_delta_t_fact), 1.2)
    new_dt = min(delta_t_fact * current_dt, max_delta_t)
    runTime.setDeltaT(new_dt)
    Info(f"deltaT = {new_dt:.6g}")


@incompressibleVoF.operation()
def increment_time(self, ctx: Context) -> None:
    """Print current simulation time and advance."""
    Info(f"Time = {ctx.runTime.timeName()}")
    ctx.runTime.increment()


@incompressibleVoF.operation(depends_on=["continuity"])
def turbulence_correction(
    self,
    turbulence: Annotated[Optional[CorrectableModel], "models"],
) -> FieldUpdates:
    """Correct two-phase turbulence model after pressure-velocity coupling."""
    if turbulence:
        turbulence.correct()
    return FieldUpdates({})


@incompressibleVoF.operation(depends_on=["turbulence_correction"])
def write_output(self, ctx: Context) -> None:
    """Write fields to disk."""
    ctx.runTime.write(True)
    ctx.runTime.printExecutionTime()
