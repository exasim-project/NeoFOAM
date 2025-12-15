# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
3-Stage Initialization Framework

This module implements a 3-stage initialization system for solvers and models:
- READ_FILES: Load configuration and data from files
- CONFIGURE: Validate and connect models (inter-model dependencies)
- SETUP: Initialize runtime structures (fields, matrices, etc.)

Usage:
    The initialization decorators are accessed via Model and Solver:

    @dataclass
    class MyModel:
        @Model.read_files
        def load_data(self):
            pass

        @Model.configure
        def connect_dependencies(self, registry):
            pass

        @Model.setup
        def initialize_fields(self, mesh):
            pass

    @dataclass
    class MySolver:
        @Solver.read_files
        def load_config(self):
            pass

        @Solver.configure
        def validate(self, registry):
            pass

        @Solver.setup
        def create_context(self, mesh):
            pass
"""

from enum import Enum
from typing import Any, Callable
from functools import wraps


# ============================================================================
# Initialization Stage Enum
# ============================================================================


class InitializationStage(str, Enum):
    """
    Enumeration of the three initialization stages.

    READ_FILES: Load configuration and data from files
    CONFIGURE: Validate and connect models (inter-model dependencies)
    SETUP: Initialize runtime structures (fields, matrices, etc.)
    """

    READ_FILES = "READ_FILES"
    CONFIGURE = "CONFIGURE"
    SETUP = "SETUP"


# ============================================================================
# Stage Decorators
# ============================================================================


def read_files(func: Callable) -> Callable:
    """
    Mark a method as belonging to READ_FILES stage.

    This decorator is typically accessed via Model.read_files or Solver.read_files.

    Methods marked with this decorator will be called during the READ_FILES
    stage of initialization, where configuration and data are loaded from files.

    Example:
        @Model.read_files
        def load_properties(self):
            self.config = load_from_file("properties.yaml")
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._init_stage = InitializationStage.READ_FILES
    return wrapper


def configure(func: Callable) -> Callable:
    """
    Mark a method as belonging to CONFIGURE stage.

    This decorator is typically accessed via Model.configure or Solver.configure.

    Methods marked with this decorator will be called during the CONFIGURE
    stage, where models can reference each other and perform validation.
    These methods receive the ModelRegistry as an argument.

    Example:
        @Model.configure
        def connect_transport(self, registry: ModelRegistry):
            self.transport = registry.get("transport")
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._init_stage = InitializationStage.CONFIGURE
    return wrapper


def setup(func: Callable) -> Callable:
    """
    Mark a method as belonging to SETUP stage.

    This decorator is typically accessed via Model.setup or Solver.setup.

    Methods marked with this decorator will be called during the SETUP
    stage, where runtime structures like fields and matrices are initialized.
    These methods receive the mesh as an argument.

    Example:
        @Model.setup
        def initialize_fields(self, mesh):
            self.velocity_field = create_field(mesh)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._init_stage = InitializationStage.SETUP
    return wrapper


# ============================================================================
# Adaptable Field
# ============================================================================


def AdaptableField(**kwargs):
    """
    Mark a field as adaptable by other models during CONFIGURE stage.

    Adaptable fields are parameters that change a model's behavior or operations.
    Other models can modify these fields during the CONFIGURE stage to select
    different implementations or algorithm variants.

    This is a wrapper around Pydantic's Field that adds 'adaptable' metadata,
    allowing the framework to identify which parameters are intended for
    inter-model configuration.

    Args:
        **kwargs: All standard Pydantic Field arguments (default, gt, ge, lt, le,
                  description, etc.)

    Returns:
        A Pydantic Field with adaptable metadata

    Example:
        class PressureAlgorithm(BaseModel):
            # Adaptable field - other models can modify this
            use_buoyancy: bool = AdaptableField(
                default=False,
                description="Use buoyancy-modified pressure equation"
            )

            # Regular field - not modifiable by other models
            tolerance: float = Field(default=1e-6, gt=0)

            # Implementation dispatch based on adaptable field
            _implementations = {
                False: StandardPressure,
                True: BuoyantPressure
            }

            def get_operations(self):
                impl = self._implementations[self.use_buoyancy]
                return impl().get_operations()

        class BuoyancyModel(BaseModel):
            @Model.configure
            def configure(self, registry: ModelRegistry):
                # Modify adaptable field in another model
                pressure = registry.get("pressure_algorithm")
                pressure.use_buoyancy = True  # Switches implementation
    """
    from pydantic import Field

    # Add adaptable metadata
    json_schema_extra = kwargs.get("json_schema_extra", {}) or {}
    json_schema_extra["adaptable"] = True
    kwargs["json_schema_extra"] = json_schema_extra

    return Field(**kwargs)


# ============================================================================
# Model Registry
# ============================================================================


class ModelRegistry:
    """
    Central registry for inter-model communication during CONFIGURE stage.

    Models are registered by name and can be retrieved by other models
    during the CONFIGURE stage to establish dependencies.
    """

    def __init__(self):
        self._models: dict[str, Any] = {}

    def register(self, name: str, model: Any) -> None:
        """
        Register a model by name.

        Args:
            name: Unique identifier for the model
            model: The model instance to register
        """
        self._models[name] = model

    def get(self, name: str) -> Any:
        """
        Get a registered model by name.

        Args:
            name: Name of the model to retrieve

        Returns:
            The model instance, or None if not found
        """
        return self._models.get(name)

    def all(self) -> dict[str, Any]:
        """
        Get all registered models.

        Returns:
            Dictionary mapping model names to model instances
        """
        return self._models.copy()

    def contains(self, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Name of the model to check

        Returns:
            True if the model is registered, False otherwise
        """
        return name in self._models

    def get_by_type(self, model_type: type) -> list[Any]:
        """
        Get all registered models of a specific type.

        Useful for working with multiple instances of the same model type
        (e.g., multiple heat sources, multiple porous zones).

        Args:
            model_type: The type/class to filter by

        Returns:
            List of all model instances of the specified type

        Example:
            heat_sources = registry.get_by_type(HeatSource)
            for source in heat_sources:
                source.enabled = False
        """
        return [
            model for model in self._models.values() if isinstance(model, model_type)
        ]

    def get_by_prefix(self, prefix: str) -> dict[str, Any]:
        """
        Get all models with names starting with a prefix.

        Useful for finding related model instances that follow a naming
        convention (e.g., "heat_source_1", "heat_source_2").

        Args:
            prefix: The prefix to match against model names

        Returns:
            Dictionary of models with matching names

        Example:
            sources = registry.get_by_prefix("heat_source_")
            for name, source in sources.items():
                print(f"{name}: {source.power}W")
        """
        return {
            name: model
            for name, model in self._models.items()
            if name.startswith(prefix)
        }

    def get_adaptable_fields(self, model_name: str) -> dict[str, Any]:
        """
        Get all adaptable fields and their current values from a model.

        Scans a model's Pydantic field definitions for fields marked with
        AdaptableField and returns their current values.

        Args:
            model_name: Name of the model to inspect

        Returns:
            Dictionary mapping adaptable field names to their current values,
            or empty dict if model not found or has no adaptable fields

        Example:
            adaptable = registry.get_adaptable_fields("pressure_algorithm")
            # Returns: {"use_buoyancy": False, "algorithm": "SIMPLE"}

            # Can check what's adaptable before modifying
            if "use_buoyancy" in adaptable:
                pressure.use_buoyancy = True
        """
        model = self.get(model_name)
        if not model:
            return {}

        # Check if model has Pydantic fields
        if not hasattr(model.__class__, "model_fields"):
            return {}

        result = {}
        for field_name, field_info in model.__class__.model_fields.items():
            # Check if field is marked as adaptable
            if field_info.json_schema_extra and field_info.json_schema_extra.get(
                "adaptable", False
            ):
                result[field_name] = getattr(model, field_name)

        return result


# ============================================================================
# Context Builder
# ============================================================================


class ContextBuilder:
    """
    Collects contributions from models and solver during SETUP stage.

    The ContextBuilder accumulates fields, models, mesh, and runtime objects
    from various sources during initialization, then builds a complete Context.
    This enables a compositional approach where multiple models can contribute
    to the simulation state.

    Example:
        builder = ContextBuilder()

        # Models contribute fields
        builder.add_field("p", pressure_field)
        builder.add_field("U", velocity_field)

        # Set infrastructure
        builder.set_mesh(mesh)
        builder.set_runtime(runTime)

        # Build final context
        ctx = builder.build()
    """

    def __init__(self):
        """Initialize an empty builder."""
        self._fields: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._mesh: Any = None
        self._runTime: Any = None

    def add_field(self, name: str, field: Any) -> None:
        """
        Add a field to the context.

        Args:
            name: Field name (e.g., "p", "U", "phi")
            field: Field object (e.g., volScalarField, volVectorField)

        Raises:
            ValueError: If field name already exists
        """
        if name in self._fields:
            raise ValueError(
                f"Field '{name}' already exists in context. Cannot add duplicate field."
            )
        self._fields[name] = field

    def add_model(self, name: str, model: Any) -> None:
        """
        Add a model to the context.

        Args:
            name: Model name (e.g., "transport", "turbulence", "algorithm")
            model: Model instance

        Raises:
            ValueError: If model name already exists
        """
        if name in self._models:
            raise ValueError(
                f"Model '{name}' already exists in context. Cannot add duplicate model."
            )
        self._models[name] = model

    def set_mesh(self, mesh: Any) -> None:
        """
        Set the mesh object.

        Args:
            mesh: The finite volume mesh (fvMesh)

        Raises:
            ValueError: If mesh has already been set
        """
        if self._mesh is not None:
            raise ValueError(
                "Mesh already set in context builder. Mesh can only be set once."
            )
        self._mesh = mesh

    def set_runtime(self, runTime: Any) -> None:
        """
        Set the runtime object.

        Args:
            runTime: The simulation time manager

        Raises:
            ValueError: If runTime has already been set
        """
        if self._runTime is not None:
            raise ValueError(
                "Runtime already set in context builder. Runtime can only be set once."
            )
        self._runTime = runTime

    def build(self) -> "Context":
        """
        Build the final Context from accumulated contributions.

        Returns:
            A complete Context object ready for simulation

        Note:
            For production code, both mesh and runTime should be set.
            For unit tests without SETUP methods, they can be left as None.
        """
        from foamadapter.framework.context import Context

        return Context(
            fields=self._fields,
            models=self._models,
            mesh=self._mesh,
            runTime=self._runTime,
        )


# ============================================================================
# Solver Initializer
# ============================================================================


class SolverInitializer:
    """
    Orchestrates 3-stage initialization for solver and its models.

    The initialization process follows three stages:
    1. READ_FILES: Load configuration and data from files
    2. CONFIGURE: Validate and connect models (with ModelRegistry)
    3. SETUP: Initialize runtime structures (with mesh)

    Within each stage, models are initialized before the solver.
    """

    def __init__(self, solver: Any):
        """
        Initialize the solver initializer.

        Args:
            solver: The solver instance to initialize
        """
        self.solver = solver
        self.registry = ModelRegistry()

    def initialize(self, mesh: Any = None) -> "Context":
        """
        Run complete 3-stage initialization and return Context.

        Args:
            mesh: Optional mesh object for SETUP stage

        Returns:
            The initialized Context ready for simulation
        """
        self._run_read_files()
        self._run_configure()
        return self._run_setup(mesh)

    def _run_read_files(self) -> None:
        """
        Execute READ_FILES stage on solver and all models.

        Models are processed first, then the solver. Each model is
        registered in the registry after its READ_FILES methods are executed.
        """
        # Models first
        for model in self._get_models():
            self._execute_stage_methods(model, InitializationStage.READ_FILES)
            # Register model for CONFIGURE stage
            model_name = getattr(model, "name", model.__class__.__name__.lower())
            self.registry.register(model_name, model)

        # Then solver
        self._execute_stage_methods(self.solver, InitializationStage.READ_FILES)

    def _run_configure(self) -> None:
        """
        Execute CONFIGURE stage - models can reference each other.

        The ModelRegistry is passed to all CONFIGURE methods, allowing
        models to find and connect to other models.
        """
        # Models first (they may depend on each other)
        for model in self._get_models():
            self._execute_stage_methods(
                model, InitializationStage.CONFIGURE, self.registry
            )

        # Then solver (can validate all models are configured)
        self._execute_stage_methods(
            self.solver, InitializationStage.CONFIGURE, self.registry
        )

    def _run_setup(self, mesh: Any) -> "Context":
        """
        Execute SETUP stage with mesh and build Context.

        Args:
            mesh: The mesh object to pass to SETUP methods

        Returns:
            The built Context with all fields and models
        """

        builder = ContextBuilder()

        # Models first - they contribute to context
        for model in self._get_models():
            self._execute_stage_methods(model, InitializationStage.SETUP, mesh, builder)

        # Then solver - finalizes context
        self._execute_stage_methods(
            self.solver, InitializationStage.SETUP, mesh, builder
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
