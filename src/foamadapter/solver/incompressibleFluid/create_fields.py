# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SimpleSolverInit - 3-stage initialization for SimpleSolver.

Implements explicit 3-stage initialization pattern:
- LOAD: Load configuration from files
- RESOLVE: Connect models and validate dependencies
- BUILD: Create lazy initializers for runtime objects

This follows the DummyInit pattern.
"""

from typing import Any, Optional
import pybFoam as pyf

from foamadapter.framework.initialization import (
    StagedInit,
    LoadResult,
    ValidationError,
    ConfigContext,
    InitializerBuilder,
    model as init_model,
)
from foamadapter.framework.initialization.lazy_init import LazyInit
from foamadapter.algorithms.pressure_velocity import PressureVelocityAlgorithm
from foamadapter.models.stability_criteria import CFLCondition
from foamadapter.models.transport_model import TransportModel
from foamadapter.models.turbulence import TurbulenceModel
from foamadapter.foam.initialization import create_time_mesh
from foamadapter.solver.incompressibleFluid.models import SimpleSolverModel


# Create the StagedInit instance
init = StagedInit("SimpleSolver")


def create_init() -> StagedInit:
    """
    Factory function for dependency injection.

    Returns the global StagedInit instance.
    """
    return init


@init.load
def load_config() -> LoadResult:
    """
    LOAD stage: Load configuration from files.

    Returns LoadResult with core_models and optional_models.
    """
    # Load fvSolution dictionary
    fv_solution = pyf.dictionary.read("system/fvSolution")

    # Create pressure-velocity algorithm
    algorithm = PressureVelocityAlgorithm.from_fvSolution(fv_solution)

    # Create CFL condition
    cfl_condition = CFLCondition()

    # Store fvSolution for later use
    algorithm._fv_solution = fv_solution

    # Detect optional models
    optional_models = SimpleSolverModel.detect_models()

    # Run LOAD on optional models
    for model in optional_models:
        if hasattr(model, "load"):
            model.load()

    return LoadResult(
        core_models=[algorithm, cfl_condition], optional_models=optional_models
    )


@init.validate_load
def validate_load_stage(core_models: list) -> list[ValidationError]:
    """Validate configuration after LOAD stage."""
    errors = []
    if not core_models or len(core_models) < 2:
        errors.append(ValidationError("core_models", "Missing core models"))
        return errors

    algorithm = core_models[0]
    if not hasattr(algorithm, "_fv_solution") or algorithm._fv_solution is None:
        errors.append(
            ValidationError("fvSolution", "Failed to read fvSolution dictionary")
        )

    return errors


@init.resolve
def resolve_models(_: list, optional_models: list, config: ConfigContext) -> None:
    """RESOLVE stage: Connect models and validate dependencies."""
    for model in optional_models:
        if hasattr(model, "resolve"):
            model.resolve(config)


@init.validate_resolve
def validate_resolve_stage(
    optional_models: list, config: Optional[ConfigContext] = None
) -> list[ValidationError]:
    """Validate model connections after RESOLVE stage."""
    errors = []
    for model in optional_models:
        if hasattr(model, "validate_stage"):
            errors.extend(model.validate_stage(config))
    return errors


@init.build
def build_lazy(core_models: list, optional_models: list) -> list[LazyInit]:
    """
    BUILD stage: Create lazy initializers for runtime objects.

    Args:
        core_models: List of core models from load stage
        optional_models: List of optional models from load stage

    Returns:
        List of LazyInit objects.
    """
    algorithm, cfl_condition = core_models
    fv_solution = algorithm._fv_solution

    builder = InitializerBuilder()

    # Build lazy initializers using current system
    time_mesh_inits = create_time_mesh(init.argv)
    builder.extend(time_mesh_inits)

    # Algorithm fields
    algo_inits = algorithm.setup()
    builder.extend(algo_inits)

    # Transport model
    builder.add(
        init_model(
            "laminarTransport",
            depends_on=["fields.U", "fields.phi"],
            create=lambda ctx: TransportModel.from_type("singlePhase").create_instance(
                ctx["fields.U"], ctx["fields.phi"]
            ),
        )
    )

    # Turbulence model
    builder.add(
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

    # Optional models: configure algorithm and add their build() LazyInits
    for model in optional_models:
        if hasattr(model, "configure_algorithm") and algorithm:
            model.configure_algorithm(algorithm)

    builder.add_optional_models(optional_models)

    # Finalize algorithm
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

        algorithm.set_pressure_reference(p, mesh, fv_solution, p_rgh)
        return algorithm

    builder.add(
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

    # CFL condition - wrap in lambda to avoid calling it during init
    builder.add_model("cfl_condition", lambda: cfl_condition)

    # Keep dictionaries alive
    builder.add_model("fvSolution_dict", lambda: fv_solution)

    # Store optional models for later access
    builder.add_model("optional_models", lambda: optional_models)

    return builder.build()
