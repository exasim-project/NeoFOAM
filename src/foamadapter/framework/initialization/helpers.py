# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Helper Functions for Lazy Initialization

Provides convenience functions for creating LazyInit objects with common patterns.
"""

from typing import Callable, Any, List, Union, Type
from .lazy_init import LazyInit


def read_vol_field(field_type: Type[Any], name: str) -> LazyInit:
    """
    Create a LazyInit for reading a volumetric field from disk.

    This is a convenience helper for the common pattern of reading OpenFOAM
    volumetric fields (volScalarField, volVectorField, etc.) from disk files.

    Args:
        field_type: The field type class (e.g., volScalarField, volVectorField)
        name: Field name (e.g., "p", "U", "T")

    Returns:
        LazyInit for reading the field from disk

    Example:
        read_vol_field(volScalarField, "p")
        # Equivalent to:
        # field("p", depends_on=["mesh"],
        #       create=lambda ctx: volScalarField.read_field(ctx["mesh"], "p"))
    """

    def create(context: dict[str, Any]) -> Any:
        mesh = context["mesh"]
        return field_type.read_field(mesh, name)

    return field(name, depends_on=["mesh"], create=create)


def field(
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: List[str] | None = None,
) -> LazyInit:
    """
    Helper for creating field lazy initializers.

    Automatically prefixes name with "fields." and sets category.

    Args:
        name: Field name (e.g., "U", "p", "nu")
        create: Function that creates the field
        depends_on: List of dependencies (default: [])

    Returns:
        LazyInit for the field

    Example:
        field("U", create=lambda ctx: create_vector_field(ctx["mesh"], U0), depends_on=["mesh"])
        # Creates: LazyInit(name="fields.U", depends_on=["mesh"], ...)
    """
    if depends_on is None:
        depends_on = []

    return LazyInit(
        name=f"fields.{name}",
        depends_on=depends_on,
        initializer=create,
        category="fields",
    )


def operator(
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: List[str] | None = None,
) -> LazyInit:
    """
    Helper for creating operator lazy initializers.

    Automatically prefixes name with "operators." and sets category.

    Args:
        name: Operator name (e.g., "momentum", "pressure_poisson")
        create: Function that creates the operator
        depends_on: List of dependencies (required)

    Returns:
        LazyInit for the operator

    Example:
        operator("momentum",
                 depends_on=["fields.U", "fields.p"],
                 create=lambda: create_momentum_equation(mesh))
    """
    if depends_on is None:
        depends_on = []

    return LazyInit(
        name=f"operators.{name}",
        depends_on=depends_on,
        initializer=create,
        category="operators",
    )


def lazy(
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: List[str] | None = None,
) -> LazyInit:
    """
    General-purpose helper for creating lazy initializers.

    Use this for objects that don't fit the field/operator categories
    (e.g., mesh, runtime, solver loops).

    Args:
        name: Object name (e.g., "mesh", "runtime", "piso_loop")
        create: Function that creates the object
        depends_on: List of dependencies (default: [])

    Returns:
        LazyInit for the object

    Example:
        lazy("mesh", create=lambda: mesh)
        lazy("piso_loop",
             depends_on=["operators.momentum", "operators.pressure_poisson"],
             create=lambda: PISOLoop())
    """
    if depends_on is None:
        depends_on = []

    return LazyInit(name=name, depends_on=depends_on, initializer=create, category=None)


def model(
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: List[str] | None = None,
) -> LazyInit:
    """
    Helper for creating model instance lazy initializers.

    Automatically prefixes name with "models." and sets category.
    Use for components like transport models, turbulence models, etc.

    Args:
        name: Model name (e.g., "transport", "turbulence", "algorithm")
        create: Function that creates the model
        depends_on: List of dependencies

    Returns:
        LazyInit for the model

    Example:
        model("transport",
              depends_on=["fields.U", "fields.phi"],
              create=lambda: singlePhaseTransportModel(U, phi))
        model("turbulence",
              depends_on=["fields.U", "fields.phi", "models.transport"],
              create=lambda: incompressibleTurbulenceModel.New(U, phi, transport))
    """
    if depends_on is None:
        depends_on = []

    return LazyInit(
        name=f"models.{name}",
        depends_on=depends_on,
        initializer=create,
        category="models",
    )


class InitializerBuilder:
    """
    Fluent builder for constructing lazy initializers.

    Provides a chainable API for building lists of LazyInit objects,
    making initialization code more readable and maintainable.

    Example:
        builder = InitializerBuilder()
        initializers = (
            builder
            .add_resource("mesh", mesh_config)
            .add_model("algorithm", algorithm)
            .add_field("U", depends_on=["mesh"], value=initial_velocity)
            .add_field("p", depends_on=["mesh"], value=lambda ctx: compute_pressure(ctx))
            .build()
        )
    """

    def __init__(self):
        self.initializers: List[LazyInit] = []

    def _normalize_lazy_init(self, li: LazyInit) -> LazyInit:
        """Normalize field name and wrap initializer for dict support."""
        import inspect

        if not li.name.startswith(("fields.", "models.", "operators.")):
            li.name = f"fields.{li.name}"

        orig_func = li.initializer
        if orig_func is None:
            return li

        # Check if function takes context parameter
        sig = inspect.signature(orig_func)
        takes_ctx = len(sig.parameters) > 0

        def wrapper(ctx=None):
            res = orig_func(ctx) if takes_ctx and ctx else orig_func()
            return res.get("value", res) if isinstance(res, dict) else res

        li.initializer = wrapper
        return li

    def add_resource(self, name: str, value: Any) -> "InitializerBuilder":
        """
        Add a top-level resource (mesh, domain, config, etc.).

        Args:
            name: Resource name
            value: The resource value (will be captured in lambda)

        Returns:
            Self for chaining
        """
        self.initializers.append(lazy(name, create=lambda: value))
        return self

    def add_model(self, name: str, value: Any) -> "InitializerBuilder":
        """
        Add a model with 'models.' prefix.

        Args:
            name: Model name (without 'models.' prefix)
            value: The model instance or callable to create it

        Returns:
            Self for chaining
        """
        if callable(value):
            self.initializers.append(model(name, create=value))
        else:
            self.initializers.append(model(name, create=lambda: value))
        return self

    def add_core_models(self, core_models: List[Any]) -> "InitializerBuilder":
        """
        Add core models with their build() LazyInit objects.

        For each core model:
        1. Adds the model instance to the models registry
        2. Calls build() if available and adds normalized LazyInit objects

        This provides a consistent interface across all solvers for adding
        core models with their associated fields and operators.

        Args:
            core_models: List of core model instances with optional names

        Returns:
            Self for chaining

        Example:
            builder.add_core_models([("algorithm", algorithm), ("core2", core_model2)])
            # Or with a list of tuples
            builder.add_core_models(core_models)
        """
        for item in core_models:
            if isinstance(item, tuple):
                name, model_instance = item
            else:
                # Use class name as default
                name = type(item).__name__.lower()
                model_instance = item

            # Add the model itself
            self.add_model(name, model_instance)

            # Add LazyInit objects from run_build() if available
            if hasattr(model_instance, "run_build"):
                lazy_inits = [
                    self._normalize_lazy_init(li) for li in model_instance.run_build()
                ]
                self.extend(lazy_inits)

        return self

    def add_field(
        self,
        name: str,
        depends_on: List[str],
        value: Union[Any, Callable[[dict[str, Any]], Any]],
    ) -> "InitializerBuilder":
        """
        Add a field with 'fields.' prefix.

        Automatically detects if value is a callable (computed field) or constant.

        Args:
            name: Field name (without 'fields.' prefix)
            depends_on: List of dependency names
            value: Constant value or callable that computes the value

        Returns:
            Self for chaining

        Example:
            .add_field("U", depends_on=["mesh"], value=initial_U)
            .add_field("p", depends_on=["mesh"], value=lambda ctx: compute_p(ctx))
        """
        if callable(value):
            self.initializers.append(field(name, depends_on=depends_on, create=value))
        else:
            self.initializers.append(
                field(name, depends_on=depends_on, create=lambda: value)
            )
        return self

    def add_operator(
        self,
        name: str,
        depends_on: List[str],
        value: Union[Any, Callable[[dict[str, Any]], Any]],
    ) -> "InitializerBuilder":
        """
        Add an operator with 'operators.' prefix.

        Args:
            name: Operator name (without 'operators.' prefix)
            depends_on: List of dependency names
            value: Operator instance or callable to create it

        Returns:
            Self for chaining
        """
        if callable(value):
            self.initializers.append(
                operator(name, depends_on=depends_on, create=value)
            )
        else:
            self.initializers.append(
                operator(name, depends_on=depends_on, create=lambda: value)
            )
        return self

    def add(self, initializer: LazyInit) -> "InitializerBuilder":
        """
        Add a pre-constructed LazyInit object.

        Useful for custom initializers or when migrating existing code.

        Args:
            initializer: LazyInit object to add

        Returns:
            Self for chaining
        """
        self.initializers.append(initializer)
        return self

    def extend(self, initializers: List[LazyInit]) -> "InitializerBuilder":
        """
        Add multiple LazyInit objects at once.

        Args:
            initializers: List of LazyInit objects

        Returns:
            Self for chaining
        """
        self.initializers.extend(initializers)
        return self

    def add_optional_models(self, optional_models: List[Any]) -> "InitializerBuilder":
        """
        Add optional models by calling their run_build() methods.

        This provides a consistent interface across all solvers for adding
        optional physics models, turbulence models, or other extensions.
        LazyInit objects from models are automatically normalized.

        Args:
            optional_models: List of optional model instances

        Returns:
            Self for chaining

        Example:
            builder.add_optional_models(optional_models)
        """
        for model in optional_models:
            if hasattr(model, "run_build"):
                lazy_inits = [self._normalize_lazy_init(li) for li in model.run_build()]
                self.extend(lazy_inits)
        return self

    def build(self) -> List[LazyInit]:
        """
        Return the constructed list of initializers.

        Returns:
            List of LazyInit objects
        """
        return self.initializers
