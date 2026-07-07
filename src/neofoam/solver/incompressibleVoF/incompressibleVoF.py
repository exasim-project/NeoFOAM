# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoF solver — interFoam-style VoF solver entrypoint.

Ported from the ``feat/incompressibleVoF`` solver_factory API to the
SolverSpec/ModelSpec API in ``stack/python_arch``. Mirrors the
``incompressibleFluid`` entrypoint with an extra ``alpha_advection`` step
at the head of the PIMPLE inner loop.

Time-loop mechanics (set_time_step with the interFoam dual Courant/alpha-Courant
adaptive dt, increment_time, write_output) stay solver-owned operations here —
they read the Foam::Time that ``create_time_mesh`` routes onto
``ctx.models["runtime"]``.
"""

from typing import Annotated, Any, Optional, Protocol

import pybFoam as pyf
import pybFoam.vof as vof
from pybFoam import (
    Info,
    surfaceScalarField,
    volScalarField,
)

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInitRunner
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.types import OperationMetadata

from .create_fields import create_init


class CorrectableModel(Protocol):
    def correct(self) -> None: ...


class TimeLoop:
    """Predicate: whether the outer time loop should continue."""

    def __call__(self, ctx: Context) -> bool:
        return bool(ctx.models["runtime"].run())


incompressibleVoF = Solver("incompressibleVoF")


@incompressibleVoF.initializer
def initialize(
    self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]
) -> Context:
    """Initialize fields and models via the staged-init runner."""
    return init.run()


@incompressibleVoF.execution_graph_step
def execution_graph(
    self: Any,
    ctx: Context,
    domain_name: Optional[str] = None,
) -> tuple[StepBuilder, Operations]:
    """Build the VoF time loop: alpha_advection -> momentum -> continuity."""
    _ = domain_name

    builder = StepBuilder()

    # Core models: alpha-advection (index 0) then PIMPLE (index 1). Each spec is
    # used as its own runtime binding for operation building.
    alpha_model = self.state.core_models[0]
    pimple_model = self.state.core_models[1]
    alpha_ops = Operations(alpha_model._build_operations_for(alpha_model))
    pimple_ops = Operations(pimple_model._build_operations_for(pimple_model))

    # Solver-owned time-loop operations.
    ops = self.operations

    time_loop_op = Operation(
        func=IterativeOp(TimeLoop()),
        metadata=OperationMetadata(op_name="time_loop"),
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

    # Optional-model operations are merged by the resolver (placed by their own
    # depends_on / operation_number).
    model_ops = Operations()
    for opt in self.state.optional_models:
        model_ops.add(opt.operations)

    return builder, model_ops


def run(
    argv: Optional[list[str]] = None,
    log_file: Optional[Any] = None,
) -> Context:
    """Run one full incompressibleVoF simulation and return the final Context.

    If ``log_file`` is given, fd 1 (stdout) is redirected to that file for the
    duration of the solve so C++ ``Info`` output ends up there.
    """
    import os
    import sys

    redirect = log_file is not None
    saved_fd: Optional[int] = None
    if redirect:
        log_path = str(log_file)
        sys.stdout.flush()
        saved_fd = os.dup(1)
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.close(log_fd)

    try:
        solver = incompressibleVoF.instantiate(argv=argv or [])
        ctx = solver.initialize()

        Info("Starting time loop")

        builder, model_ops = solver.execution_graph(ctx=ctx)

        resolver = DAGResolver()
        resolved = resolver.resolve(builder, model_ops)
        resolved.operations.run(ctx)

        Info("End")
        return ctx
    finally:
        if redirect and saved_fd is not None:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)


# ---------------------------------------------------------------------------
# Solver-owned time-loop operations
# ---------------------------------------------------------------------------


@incompressibleVoF.operation()
def set_time_step(
    ctx: Context,
    phi: surfaceScalarField,
    alpha1: volScalarField,
    runtime: Annotated[Any, "models"],
) -> None:
    """Adjust the time step from both flow-CFL and alpha-CFL (interFoam setDeltaT)."""
    # adjustTimeStep / maxCo / maxAlphaCo / maxDeltaT are static controlDict
    # entries — read them from the file (Time.controlDict() is not bound).
    ctrl_dict = pyf.dictionary.read("system/controlDict")

    def _dict_bool(key: str, default: bool) -> bool:
        if ctrl_dict.found(key):
            try:
                return bool(ctrl_dict.getOrDefault[bool](key, default))
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

    # Flow + interface (alpha) Courant numbers.
    maxCoNum, meanCoNum = pyf.computeCFLNumber(phi)
    alphaCoNum, meanAlphaCo = vof.computeAlphaCourantNumber(phi, alpha1)

    Info(f"Courant Number mean: {meanCoNum:.4g}  max: {maxCoNum:.4g}")
    Info(f"Interface Courant Number mean: {meanAlphaCo:.4g}  max: {alphaCoNum:.4g}")

    # Mirror interFoam setDeltaT.H: limit by both Co constraints, allow a ramped
    # increase (at most +20% per step) when Co is well below the limit.
    current_dt = runtime.deltaTValue()
    max_delta_t_fact = min(
        max_co / (maxCoNum + 1e-15),
        max_alpha_co / (alphaCoNum + 1e-15),
    )
    delta_t_fact = min(min(max_delta_t_fact, 1.0 + 0.1 * max_delta_t_fact), 1.2)
    new_dt = min(delta_t_fact * current_dt, max_delta_t)
    runtime.setDeltaT(new_dt)
    Info(f"deltaT = {new_dt:.6g}")


@incompressibleVoF.operation()
def increment_time(runtime: Annotated[Any, "models"]) -> None:
    """Print current simulation time and advance."""
    Info(f"Time = {runtime.timeName()}")
    runtime.increment()


@incompressibleVoF.operation(depends_on=["continuity"])
def turbulence_correction(
    turbulence: Annotated[Optional[CorrectableModel], "models"],
) -> FieldUpdates:
    """Correct the two-phase turbulence model after pressure-velocity coupling."""
    if turbulence:
        turbulence.correct()
    return FieldUpdates({})


@incompressibleVoF.operation(depends_on=["turbulence_correction"])
def write_output(runtime: Annotated[Any, "models"]) -> None:
    """Write fields to disk and report execution time."""
    runtime.write(True)
    runtime.printExecutionTime()
