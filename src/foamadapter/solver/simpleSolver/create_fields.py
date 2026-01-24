# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SimpleSolverInit - FastAPI-style initialization syntax.

This demonstrates the FastAPI-style pattern:
    init = Init("name")

    @init.step
    def runTime() -> pyf.Time: ...

    @init.step
    def mesh(runTime: Annotated[Time, Depends(runTime)]) -> Mesh: ...
"""

from typing import Any, Annotated
import pybFoam as pyf

from foamadapter.framework.context import Context
from foamadapter.framework.initialization import Init, Depends, execute_initialization
from foamadapter.algorithms.pressure_velocity import PressureVelocityAlgorithm
from foamadapter.models.stability_criteria import CFLCondition
from foamadapter.models.transport_model import TransportModel
from foamadapter.models.turbulence import TurbulenceModel
from foamadapter.foam.initialization import create_time_mesh
from foamadapter.framework.initialization.helpers import lazy, model as init_model
from foamadapter.solver.simpleSolver.models import SimpleSolverModel


init = Init("SimpleSolver")


@init.step
def optional_models() -> list[Any]:
    """Detect and instantiate optional models."""
    return SimpleSolverModel.detect_models()


@init.step
def fvSolution() -> Any:
    """Load fvSolution dictionary."""
    return pyf.dictionary.read("system/fvSolution")


@init.step
def algorithm(
    fvSolution_dict: Annotated[Any, Depends(fvSolution)],
) -> PressureVelocityAlgorithm:
    """
    Create pressure-velocity algorithm.

    Depends on fvSolution being loaded first (FastAPI-style).
    """
    return PressureVelocityAlgorithm.from_fvSolution(fvSolution_dict)


@init.step
def cfl_condition() -> CFLCondition:
    """Create CFL condition (no dependencies)."""
    return CFLCondition()


# ============================================================================
# BUILD CONTEXT (combines init steps into Context)
# ============================================================================


@init.build_context
def build() -> Context:
    """
    Build context using current lazy initialization system.

    This bridges between the FastAPI-style Init pattern and
    the current initialization system.
    """
    # Get argv from init instance
    argv = init.argv

    # Get algorithm and cfl_condition (cached via @init.step)
    algo = algorithm()
    cfl = cfl_condition()
    fv_solution = fvSolution()

    # Detect optional models (NEW)
    models = optional_models()
    if models:
        from pybFoam import Info

        Info(f"Detected {len(models)} optional model(s): {[m.name for m in models]}")

    # Build lazy initializers (current system)
    initializers = create_time_mesh(argv)

    # Algorithm fields
    initializers.extend(algo.setup())

    # Transport model
    initializers.append(
        init_model(
            "laminarTransport",
            depends_on=["fields.U", "fields.phi"],
            create=lambda ctx: TransportModel.from_type("singlePhase").create_instance(
                ctx["fields.U"], ctx["fields.phi"]
            ),
        )
    )

    # Turbulence model
    initializers.append(
        init_model(
            "turbulence",
            depends_on=["fields.U", "fields.phi", "models.laminarTransport"],
            create=lambda ctx: TurbulenceModel.from_type(
                "openfoam_rts"
            ).create_instance(
                ctx["fields.U"], ctx["fields.phi"], ctx["models.laminarTransport"]
            ),
        )
    )

    # Optional models: Add their fields and models (NEW)
    for model in models:
        if hasattr(model, "build"):
            initializers.extend(model.build())

    # Final algorithm build
    def finalize_algorithm(context: dict[str, Any]) -> Any:
        p = context["fields.p"]
        mesh = context["mesh"]

        # Check if p_rgh exists (Boussinesq mode)
        p_rgh = None
        fields_dict = context.get("fields", {})
        if isinstance(fields_dict, dict) and "p_rgh" in fields_dict:
            p_rgh = fields_dict["p_rgh"]
        elif hasattr(fields_dict, "p_rgh"):
            p_rgh = fields_dict.p_rgh

        # Configure algorithm for optional models (e.g., Boussinesq)
        for model in models:
            if hasattr(model, "configure_algorithm"):
                model.configure_algorithm(algo)

        algo.set_pressure_reference(p, mesh, fv_solution, p_rgh)
        return algo

    initializers.append(
        init_model(
            "algorithm",
            depends_on=[
                "fields.p",
                "mesh",
                "models.laminarTransport",
                "models.turbulence",
            ],
            create=finalize_algorithm,
        )
    )

    # CFL condition
    initializers.append(lazy("cfl_condition", create=lambda _: cfl))

    # Keep dictionaries alive
    initializers.append(lazy("fvSolution_dict", create=lambda _: fv_solution))

    # Store optional models for later access (NEW)
    initializers.append(lazy("optional_models", create=lambda _: models))

    # Execute initialization
    ctx = execute_initialization(initializers)

    return ctx
