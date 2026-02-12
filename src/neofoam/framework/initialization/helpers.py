# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Helper Functions for Lazy Initialization

Provides convenience functions for creating LazyInit objects with common patterns.
"""

from typing import Callable, Any, List, Optional, Union, Type
from dataclasses import replace as _replace
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


# ---------------------------------------------------------------------------
# Core factory — all four helpers delegate to this
# ---------------------------------------------------------------------------


def _make_lazy(
    prefix: Optional[str],
    category: Optional[str],
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: Optional[List[str]] = None,
) -> LazyInit:
    """Internal factory shared by field / operator / model / lazy."""
    full_name = f"{prefix}.{name}" if prefix else name
    return LazyInit(
        name=full_name,
        depends_on=depends_on or [],
        initializer=create,
        category=category,
    )


def field(
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: Optional[List[str]] = None,
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
    """
    return _make_lazy("fields", "fields", name, create, depends_on)


def operator(
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: Optional[List[str]] = None,
) -> LazyInit:
    """
    Helper for creating operator lazy initializers.

    Automatically prefixes name with "operators." and sets category.

    Args:
        name: Operator name (e.g., "momentum", "pressure_poisson")
        create: Function that creates the operator
        depends_on: List of dependencies

    Returns:
        LazyInit for the operator

    Example:
        operator("momentum", depends_on=["fields.U", "fields.p"], create=lambda: ...)
    """
    return _make_lazy("operators", "operators", name, create, depends_on)


def lazy(
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: Optional[List[str]] = None,
) -> LazyInit:
    """
    General-purpose helper for creating lazy initializers.

    Use this for objects that don't fit the field/operator/model categories
    (e.g., mesh, runtime, solver loops).

    Args:
        name: Object name (e.g., "mesh", "runtime", "piso_loop")
        create: Function that creates the object
        depends_on: List of dependencies (default: [])

    Returns:
        LazyInit for the object

    Example:
        lazy("mesh", create=lambda: mesh)
    """
    return _make_lazy(None, None, name, create, depends_on)


def model(
    name: str,
    create: Union[Callable[[], Any], Callable[[dict[str, Any]], Any]],
    depends_on: Optional[List[str]] = None,
) -> LazyInit:
    """
    Helper for creating model instance lazy initializers.

    Automatically prefixes name with "models." and sets category.

    Args:
        name: Model name (e.g., "transport", "turbulence", "algorithm")
        create: Function that creates the model
        depends_on: List of dependencies

    Returns:
        LazyInit for the model

    Example:
        model("transport", depends_on=["fields.U"], create=lambda: ...)
    """
    return _make_lazy("models", "models", name, create, depends_on)


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

    @staticmethod
    def _normalize_lazy_init(li: LazyInit) -> LazyInit:
        """Return a *new* LazyInit with a normalized name and wrapped initializer.

        * Adds a ``fields.`` prefix when the name has no recognised prefix.
        * Wraps the initializer so that dict results with a ``value`` key are
          automatically unwrapped.

        The original ``li`` is **never mutated**.
        """
        import inspect

        new_name = li.name
        if not new_name.startswith(("fields.", "models.", "operators.")):
            new_name = f"fields.{new_name}"

        orig_func = li.initializer
        if orig_func is None:
            return _replace(li, name=new_name)

        # Check if function takes context parameter
        sig = inspect.signature(orig_func)
        takes_ctx = len(sig.parameters) > 0

        def wrapper(ctx=None):
            res = orig_func(ctx) if takes_ctx and ctx else orig_func()
            return res.get("value", res) if isinstance(res, dict) else res

        return _replace(li, name=new_name, initializer=wrapper)

    # ---- private helper: collapses add_field / add_model / add_operator ----

    def _add_typed(
        self,
        factory: Callable[..., LazyInit],
        name: str,
        value: Any,
        depends_on: Optional[List[str]] = None,
    ) -> "InitializerBuilder":
        """Shared logic for add_field / add_model / add_operator."""
        if callable(value):
            self.initializers.append(factory(name, create=value, depends_on=depends_on))
        else:
            # Capture by closure — each _add_typed call has its own scope
            self.initializers.append(
                factory(name, create=lambda: value, depends_on=depends_on)
            )
        return self

    # ---- public API ----

    def add_resource(self, name: str, value: Any) -> "InitializerBuilder":
        """
        Add a top-level resource (mesh, domain, config, etc.).

        Args:
            name: Resource name
            value: The resource value (will be captured in lambda)

        Returns:
            Self for chaining
        """
        # Wrap in a zero-arg closure; default-arg capture avoids late-binding (P-2.5)
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
        return self._add_typed(model, name, value)

    def add_core_models(self, core_models: List[Any]) -> "InitializerBuilder":
        """
        Add core models with their build() LazyInit objects.

        For each core model:
        1. Adds the model instance to the models registry
        2. Calls build() if available and adds normalized LazyInit objects

        Args:
            core_models: List of core model instances with optional names

        Returns:
            Self for chaining

        Example:
            builder.add_core_models([("algorithm", algorithm), ("core2", core_model2)])
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

        Args:
            name: Field name (without 'fields.' prefix)
            depends_on: List of dependency names
            value: Constant value or callable that computes the value

        Returns:
            Self for chaining
        """
        return self._add_typed(field, name, value, depends_on)

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
        return self._add_typed(operator, name, value, depends_on)

    def add(self, initializer: LazyInit) -> "InitializerBuilder":
        """
        Add a pre-constructed LazyInit object.

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

        Args:
            optional_models: List of optional model instances

        Returns:
            Self for chaining
        """
        for m in optional_models:
            if hasattr(m, "run_build"):
                lazy_inits = [self._normalize_lazy_init(li) for li in m.run_build()]
                self.extend(lazy_inits)
        return self

    def build(self) -> List[LazyInit]:
        """
        Return the constructed list of initializers.

        Returns:
            List of LazyInit objects
        """
        return self.initializers
