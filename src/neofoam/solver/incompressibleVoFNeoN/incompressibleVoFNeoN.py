# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""incompressibleVoFNeoN: laminar interFoam on NeoN, surface tension, fixed or adaptive deltaT."""

import os
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

from neofoam.framework.context import Context
from neofoam.framework.graph import DAGResolver
from neofoam.framework.initialization import Depends, StagedInitRunner
from neofoam.framework.model import ModelRuntime
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.solver import Solver
from neofoam.framework.tools import PreprocessConfig
from neofoam.framework.types import OperationMetadata
from neofoam.preprocess import SetFieldsConfig
from neofoam.solver.incompressibleFluidNeoN.models.solution_loop import SolutionLoopPredicate
from neofoam.solver.incompressibleVoF.configs import (
    ControlDictConfig,
    GravityConfig,
    TransportPropertiesConfig,
    TurbulencePropertiesConfig,
)
from neofoam.solver.neon_runtime import NeoNControlConfig, ensure_neon_initialized
from neofoam.tools.block_mesh import BlockMeshDictConfig
from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig

from .create_fields import create_init
from .models.alpha_advection import advectionModelNeoN
from .models.pressure_velocity import PressureVelocityAlgorithmVoFNeoN
from .models.surface_forces import SurfaceForcesVoFNeoN

incompressibleVoFNeoN = Solver("incompressibleVoFNeoN")

# The case-authoring schema of incompressibleVoF, plus the NeoN executor key.
incompressibleVoFNeoN.config(ControlDictConfig)
incompressibleVoFNeoN.config(NeoNControlConfig)
incompressibleVoFNeoN.config(PreprocessConfig)
incompressibleVoFNeoN.config(BlockMeshDictConfig)
incompressibleVoFNeoN.config(SnappyHexMeshDictConfig)
incompressibleVoFNeoN.config(TransportPropertiesConfig)
incompressibleVoFNeoN.config(GravityConfig)
incompressibleVoFNeoN.config(TurbulencePropertiesConfig)
incompressibleVoFNeoN.config(SetFieldsConfig)

incompressibleVoFNeoN.models(advectionModelNeoN, required=True)
incompressibleVoFNeoN.models(PressureVelocityAlgorithmVoFNeoN, required=True)
incompressibleVoFNeoN.models(SurfaceForcesVoFNeoN, required=True)


def _report_time_step(ctx: Context) -> None:
    """setDeltaT.H's ``deltaT = ...`` line, at full precision."""
    print(f"deltaT = {ctx.models['solution_loop'].current_delta_t()!r}", flush=True)


def _core_model(state: Any, spec_name: str) -> Any:
    """An instantiated core model by its spec name."""
    return next(
        member
        for member in state.core_models
        if isinstance(member, ModelRuntime) and member.spec.name == spec_name
    )


@incompressibleVoFNeoN.initializer
def initialize(self: Any, init: Annotated[StagedInitRunner, Depends(create_init)]) -> Context:
    """Initialize fields and models via the staged-init runner."""
    return init.run()


@incompressibleVoFNeoN.execution_graph_step
def execution_graph(
    self: Any, ctx: Context, domain_name: Optional[str] = None
) -> tuple[StepBuilder, Operations]:
    """The time loop: alpha_advection -> surface_forces -> momentum -> continuity -> write."""
    _ = domain_name
    alpha_model, pimple_model, forces_model = self.state.core_models[:3]
    alpha_ops = Operations(alpha_model.build_operations_for(alpha_model))
    forces_ops = Operations(forces_model.build_operations_for(forces_model))
    pimple_ops = Operations(pimple_model.build_operations_for(pimple_model))
    loop_ops = Operations(_core_model(self.state, "solutionLoop").operations)
    writer_ops = Operations(_core_model(self.state, "fieldWriter").operations)

    builder = StepBuilder()
    time_loop_op = Operation(
        func=IterativeOp(SolutionLoopPredicate()),
        metadata=OperationMetadata(op_name="time_loop"),
    )
    with builder.loop(time_loop_op) as time_builder:
        time_builder.step(loop_ops["set_time_step"])
        time_builder.step(
            Operation(
                func=SequentialOp(_report_time_step),
                metadata=OperationMetadata(op_name="report_time_step"),
            )
        )
        time_builder.step(loop_ops["increment_time"])
        time_builder.step(alpha_ops["alpha_advection"])
        time_builder.step(forces_ops["surface_forces"])
        time_builder.step(pimple_ops["momentum"])
        time_builder.step(pimple_ops["continuity"])
        time_builder.step(writer_ops["write_output"])
    return builder, Operations()


def run(argv: Optional[list[str]] = None, log_file: Optional[Any] = None) -> Context:
    """Run one incompressibleVoFNeoN simulation in the working directory; return the Context.

    Use it for laminar MULES cases with a fixed or adaptive deltaT and constant surface
    tension on Serial, CPU or GPU; ``neofoam.solver.incompressibleVoF`` runs the other
    cases on pybFoam.

    Kokkos can be initialized once per process, so run one case per process. With
    ``log_file``, fd 1 is redirected to it for the duration of the solve. Boundary conditions
    are NeoN's: ``totalPressure`` without a flux/velocity context is a fixedValue at ``p0``
    and ``pressureInletOutletVelocity`` is zeroGradient.

    Example::

        run(["incompressibleVoFNeoN"])
    """
    ensure_neon_initialized(list(argv) if argv else ["incompressibleVoFNeoN"])
    saved_fd: Optional[int] = None
    if log_file is not None:
        sys.stdout.flush()
        saved_fd = os.dup(1)
        log_fd = os.open(str(Path(str(log_file))), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.close(log_fd)
    try:
        solver = incompressibleVoFNeoN.instantiate(argv=argv or [])
        ctx = solver.initialize()
        builder, model_ops = solver.execution_graph(ctx=ctx)
        DAGResolver().resolve(builder, model_ops).operations.run(ctx)
        print("End")
        return ctx
    finally:
        if saved_fd is not None:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
