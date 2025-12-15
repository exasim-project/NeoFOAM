# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Solver Initializer

Orchestrates the 3-stage initialization process for solvers and their models.
"""

from typing import Any

from .stages import InitializationStage
from .config_context import ConfigContext
from .context_builder import ContextBuilder


class SolverInitializer:
    """
    Orchestrates 3-stage initialization for solver and its models.

    The initialization process follows three stages:
    1. LOAD: Load configuration and data from files
    2. RESOLVE_DEPENDENCIES: Validate and connect models (with ConfigContext)
    3. BUILD: Initialize runtime structures (with mesh)

    Within each stage, models are initialized before the solver.
    """

    def __init__(self, solver: Any, region: str = "default"):
        """
        Initialize the solver initializer.

        Args:
            solver: The solver instance to initialize
            region: Name of the region for this solver (default: "default")
        """
        self.solver = solver
        self.region = region
        self.config = ConfigContext(current_region=region)

    def initialize(self, mesh: Any = None) -> "Context":
        """
        Run complete 3-stage initialization and return Context.

        Args:
            mesh: Optional mesh object for BUILD stage

        Returns:
            The initialized Context ready for simulation
        """
        self._run_load()
        self._run_resolve_dependencies()
        return self._run_build(mesh)

    def _run_load(self) -> None:
        """
        Execute LOAD stage on solver and all models.

        Models are processed first, then the solver. Each model is
        registered in the config context after its LOAD methods are executed.
        """
        # Models first
        for model in self._get_models():
            self._execute_stage_methods(model, InitializationStage.LOAD)
            # Register model for RESOLVE_DEPENDENCIES stage
            model_name = getattr(model, "name", model.__class__.__name__.lower())
            self.config.register(model_name, model)

        # Then solver
        self._execute_stage_methods(self.solver, InitializationStage.LOAD)

    def _run_resolve_dependencies(self) -> None:
        """
        Execute RESOLVE_DEPENDENCIES stage - models can reference each other.

        The ConfigContext is passed to all RESOLVE_DEPENDENCIES methods, allowing
        models to find and connect to other models.
        """
        # Models first (they may depend on each other)
        for model in self._get_models():
            self._execute_stage_methods(
                model, InitializationStage.RESOLVE_DEPENDENCIES, self.config
            )

        # Then solver (can validate all models are configured)
        self._execute_stage_methods(
            self.solver, InitializationStage.RESOLVE_DEPENDENCIES, self.config
        )

    def _run_build(self, mesh: Any) -> "Context":
        """
        Execute BUILD stage with mesh and build Context.

        Args:
            mesh: The mesh object to pass to BUILD methods

        Returns:
            The built Context with all fields and models
        """

        builder = ContextBuilder()

        # Models first - they contribute to context
        for model in self._get_models():
            self._execute_stage_methods(model, InitializationStage.BUILD, mesh, builder)

        # Then solver - finalizes context
        self._execute_stage_methods(
            self.solver, InitializationStage.BUILD, mesh, builder
        )

        # Build and return the Context
        return builder.build()

    def _get_models(self) -> list[Any]:
        """
        Get all models from the solver.

        Looks for a get_models() method on the solver, or falls back
        to collecting all attributes that have a 'name' attribute.

        Returns:
            List of model instances
        """
        # Try get_models() method first
        if hasattr(self.solver, "get_models") and callable(self.solver.get_models):
            return self.solver.get_models()

        # Fallback: collect attributes with 'name' attribute
        models = []
        for attr_name in dir(self.solver):
            if attr_name.startswith("_"):
                continue
            # Skip Pydantic internal attributes to avoid deprecation warnings
            if attr_name in ("model_fields", "model_computed_fields", "model_config"):
                continue
            attr = getattr(self.solver, attr_name, None)
            if attr is not None and hasattr(attr, "name") and not callable(attr):
                models.append(attr)

        return models

    def _execute_stage_methods(
        self, obj: Any, stage: InitializationStage, *args
    ) -> None:
        """
        Execute all methods marked with given stage decorator.

        Args:
            obj: The object (solver or model) to execute methods on
            stage: The InitializationStage to execute
            *args: Arguments to pass to the stage methods
        """
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            # Skip Pydantic internal attributes to avoid deprecation warnings
            if attr_name in ("model_fields", "model_computed_fields", "model_config"):
                continue

            attr = getattr(obj, attr_name, None)
            if callable(attr) and hasattr(attr, "_init_stage"):
                if attr._init_stage == stage:
                    attr(*args)
