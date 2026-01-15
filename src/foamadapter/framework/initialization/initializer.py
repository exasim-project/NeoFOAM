# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Solver Initializer

Orchestrates the 3-stage initialization process for solvers and their models.
"""

from typing import Any
import networkx as nx

from .stages import InitializationStage
from .config_context import ConfigContext
from .lazy_init import LazyInit
from ..context import Context


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

        LOAD methods MUST return dict[str, Any] with config items to register.
        Empty dict {} is valid for models with no configurable objects.
        """
        # Models first
        for model in self._get_models():
            # Execute LOAD methods and collect return values
            config_items = self._execute_stage_methods_with_return(
                model, InitializationStage.LOAD
            )

            # Register model itself for RESOLVE_DEPENDENCIES stage
            model_name = getattr(model, "name", model.__class__.__name__.lower())
            self.config.register(model_name, model)

            # Register config items returned by LOAD methods
            self._register_config_items(config_items, model_name)

        # Then solver
        config_items = self._execute_stage_methods_with_return(
            self.solver, InitializationStage.LOAD
        )
        self._register_config_items(config_items, "solver")

    def _register_config_items(self, config_items: list[Any], source: str) -> None:
        """
        Register config items returned from LOAD stage.

        Expects list of dict[str, Any] where each dict contains config items.
        Each key-value pair is registered in ConfigContext.

        Args:
            config_items: List of dicts returned by LOAD methods
            source: Name of model/solver for error messages
        """
        if not config_items:
            return

        for item in config_items:
            if not isinstance(item, dict):
                raise TypeError(
                    f"LOAD methods must return dict[str, Any]. "
                    f"{source} returned {type(item).__name__}"
                )

            for key, value in item.items():
                print(f"DEBUG: Registering config item '{key}' from {source}")
                self.config.register(key, value)

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
        Execute BUILD stage with lazy initialization and DAG resolution.

        1. Collect lazy initializers from models and solver
        2. Resolve dependencies using topological sort (DAG)
        3. Execute initializers in dependency order
        4. Build and return Context

        Args:
            mesh: The mesh object to pass to BUILD methods

        Returns:
            The built Context with all fields and models
        """
        all_lazy_inits = []

        # Collect lazy initializers from models
        for model in self._get_models():
            result = self._execute_stage_methods_with_return(
                model, InitializationStage.BUILD, mesh
            )
            if result is not None:
                if isinstance(result, list):
                    all_lazy_inits.extend(result)
                elif isinstance(result, LazyInit):
                    all_lazy_inits.append(result)

        # Collect lazy initializers from solver
        result = self._execute_stage_methods_with_return(
            self.solver, InitializationStage.BUILD, mesh
        )
        if result is not None:
            if isinstance(result, list):
                all_lazy_inits.extend(result)
            elif isinstance(result, LazyInit):
                all_lazy_inits.append(result)

        # Resolve dependencies using DAG (topological sort)
        sorted_inits = self._topological_sort(all_lazy_inits)

        # Execute in dependency order and collect results
        initialized_objects: dict[str, Any] = {}
        for lazy_init in sorted_inits:
            # Pass initialized_objects as context to lazy initializers
            obj = lazy_init.execute(context=initialized_objects)
            initialized_objects[lazy_init.name] = obj

        # Build Context from initialized objects
        return self._build_context_from_objects(initialized_objects)

    def _topological_sort(self, lazy_inits: list[LazyInit]) -> list[LazyInit]:
        """
        Sort lazy initializers by dependencies using networkx topological sort.

        Args:
            lazy_inits: List of LazyInit objects to sort

        Returns:
            List of LazyInit objects in dependency order

        Raises:
            ValueError: If a dependency references a non-existent initializer
            nx.NetworkXError: If circular dependencies are detected
        """
        # Create name -> LazyInit mapping
        name_to_init = {li.name: li for li in lazy_inits}

        # Validate all dependencies exist
        for li in lazy_inits:
            for dep in li.depends_on:
                if dep not in name_to_init:
                    raise ValueError(
                        f"LazyInit '{li.name}' depends on '{dep}', "
                        f"but '{dep}' was not found in lazy initializers"
                    )

        # Build DAG
        G = nx.DiGraph()
        for li in lazy_inits:
            G.add_node(li.name, lazy_init=li)
            for dep in li.depends_on:
                G.add_edge(dep, li.name)

        # Topological sort with cycle detection
        try:
            sorted_names = list(nx.lexicographical_topological_sort(G))
        except nx.NetworkXError as e:
            # Find cycle for better error message
            try:
                cycle = nx.find_cycle(G)
                cycle_names = [edge[0] for edge in cycle]
                raise ValueError(
                    f"Circular dependency detected: {' -> '.join(cycle_names)}"
                ) from e
            except nx.NetworkXNoCycle:
                raise ValueError(f"DAG error: {str(e)}") from e

        # Return LazyInit objects in sorted order
        return [name_to_init[name] for name in sorted_names]

    def _build_context_from_objects(self, objects: dict[str, Any]) -> "Context":
        """
        Build Context from initialized objects.

        Categorizes objects into fields, models, mesh, runtime based on
        their names or categories.

        Args:
            objects: Dictionary mapping names to initialized objects

        Returns:
            Context with categorized objects
        """
        from ..context import Context

        fields = {}
        models = {}
        mesh = None
        runtime = None

        for name, obj in objects.items():
            if name.startswith("fields."):
                field_name = name.replace("fields.", "")
                fields[field_name] = obj
            elif name.startswith("operators."):
                # Operators stored as models
                operator_name = name.replace("operators.", "")
                models[operator_name] = obj
            elif name.startswith("models."):
                # Models stored with their base name
                model_name = name.replace("models.", "")
                models[model_name] = obj
            elif name == "mesh":
                mesh = obj
            elif name == "runtime":
                runtime = obj
            else:
                # Store other objects as models
                models[name] = obj

        return Context(fields=fields, models=models, mesh=mesh, runTime=runtime)

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
            return self.solver.get_models()  # type: ignore[no-any-return]

        # Fallback: collect attributes with 'name' attribute
        models: list[Any] = []
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
        self, obj: Any, stage: InitializationStage, *args: Any
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

    def _execute_stage_methods_with_return(
        self, obj: Any, stage: InitializationStage, *args: Any
    ) -> Any:
        """
        Execute stage methods and collect return values.

        Similar to _execute_stage_methods but returns collected values
        from stage methods (for BUILD stage).

        Args:
            obj: The object (solver or model) to execute methods on
            stage: The InitializationStage to execute
            *args: Arguments to pass to the stage methods

        Returns:
            Flattened list of returned values (LazyInit objects)
        """
        results = []
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            # Skip Pydantic internal attributes
            if attr_name in ("model_fields", "model_computed_fields", "model_config"):
                continue

            attr = getattr(obj, attr_name, None)
            if callable(attr) and hasattr(attr, "_init_stage"):
                if attr._init_stage == stage:
                    result = attr(*args)
                    if result is not None:
                        results.append(result)

        # Flatten list of LazyInit objects
        flat_results = []
        for result in results:
            if isinstance(result, list):
                flat_results.extend(result)
            else:
                flat_results.append(result)

        return flat_results if flat_results else None
