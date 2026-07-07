# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Helper Functions for Lazy Initialization

Provides convenience functions for creating InitStep objects with common patterns.
"""

from typing import Callable, Any, List, Optional, Protocol, Union, runtime_checkable
from .init_step import InitStep, InitCategory


@runtime_checkable
class BuildsInitSteps(Protocol):
    """Protocol matched by objects that contribute :class:`InitStep` lists.

    Models / sub-systems implement ``run_build()`` to expose their
    initialization steps to :class:`InitializerBuilder`. The Protocol gives
    the contract a name (mypy can check it) and replaces the duck-typed
    ``hasattr(item, "run_build")`` checks used elsewhere.
    """

    def run_build(self) -> List[InitStep]: ...


def _make_lazy(
    prefix: Optional[str],
    category: InitCategory,
    name: str,
    create: Callable[[dict[str, Any]], Any],
    depends_on: Optional[List[str]] = None,
    write: bool = False,
    replaces: Optional[List[str]] = None,
) -> InitStep:
    """Internal factory shared by field / operator / model / lazy."""
    full_name = f"{prefix}.{name}" if prefix else name
    return InitStep(
        name=full_name,
        depends_on=depends_on or [],
        initializer=create,
        category=category,
        write=write,
        replaces=replaces or [],
    )


def field(
    name: str,
    create: Callable[[dict[str, Any]], Any],
    depends_on: Optional[List[str]] = None,
    write: bool = False,
    *,
    replaces: Optional[List[str]] = None,
) -> InitStep:
    """
    Helper for creating field lazy initializers.

    Automatically prefixes name with "fields." and sets category.

    Args:
        name: Field name (e.g., "U", "p", "nu")
        create: Function that creates the field
        depends_on: List of dependencies (default: [])
        write: Flag this field for persistence (auto-write). Collected into
            ``Context.write_fields`` so a per-field write backend knows which
            fields to write to disk.

    Returns:
        InitStep for the field

    Example:
        field("U", create=lambda ctx: create_vector_field(ctx["mesh"], U0),
              depends_on=["mesh"], write=True)
    """
    return _make_lazy(
        "fields", "fields", name, create, depends_on, write=write, replaces=replaces
    )


def operator(
    name: str,
    create: Callable[[dict[str, Any]], Any],
    depends_on: Optional[List[str]] = None,
) -> InitStep:
    """
    Helper for creating operator lazy initializers.

    Automatically prefixes name with "operators." and sets category.

    Args:
        name: Operator name (e.g., "momentum", "pressure_poisson")
        create: Function that creates the operator
        depends_on: List of dependencies

    Returns:
        InitStep for the operator

    Example:
        operator("momentum", depends_on=["fields.U", "fields.p"], create=lambda ctx: ...)
    """
    return _make_lazy("operators", "operators", name, create, depends_on)


def lazy(
    name: str,
    create: Callable[[dict[str, Any]], Any],
    depends_on: Optional[List[str]] = None,
    *,
    replaces: Optional[List[str]] = None,
) -> InitStep:
    """
    General-purpose helper for creating lazy initializers.

    Use this for objects that don't fit the field/operator/model categories
    (e.g., mesh, runtime, solver loops).

    Args:
        name: Object name (e.g., "mesh", "runtime", "piso_loop")
        create: Function that creates the object
        depends_on: List of dependencies (default: [])
        replaces: Default-step names this step supersedes. When set, the
            builder drops any other step carrying one of these names so a
            generated resource can stand in for the default one.

    Returns:
        InitStep for the object

    Example:
        lazy("mesh", create=lambda ctx: mesh)
    """
    return _make_lazy(None, "resource", name, create, depends_on, replaces=replaces)


def model(
    name: str,
    create: Callable[[dict[str, Any]], Any],
    depends_on: Optional[List[str]] = None,
) -> InitStep:
    """
    Helper for creating model instance lazy initializers.

    Automatically prefixes name with "models." and sets category.

    Args:
        name: Model name (e.g., "transport", "turbulence", "algorithm")
        create: Function that creates the model
        depends_on: List of dependencies

    Returns:
        InitStep for the model

    Example:
        model("transport", depends_on=["fields.U"], create=lambda ctx: ...)
    """
    return _make_lazy("models", "models", name, create, depends_on)


class InitializerBuilder:
    """
    Fluent builder for constructing lazy initializers.

    Provides a chainable API for building lists of InitStep objects,
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

    def __init__(self) -> None:
        self.initializers: List[InitStep] = []

    # ---- private helper: collapses add_field / add_model / add_operator ----

    def _add_typed(
        self,
        factory: Callable[..., InitStep],
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
                factory(name, create=lambda _ctx: value, depends_on=depends_on)
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
        self.initializers.append(lazy(name, create=lambda _ctx: value))
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
        Add core models with their build() InitStep objects.

        For each core model:
        1. Adds the model instance to the models registry
        2. Calls build() if available and adds normalized InitStep objects

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

            # Add InitStep objects from run_build() if available.
            if isinstance(model_instance, BuildsInitSteps):
                self.extend(model_instance.run_build())

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

    def add(self, initializer: InitStep) -> "InitializerBuilder":
        """
        Add a pre-constructed InitStep object.

        Args:
            initializer: InitStep object to add

        Returns:
            Self for chaining
        """
        self.initializers.append(initializer)
        return self

    def extend(self, initializers: List[InitStep]) -> "InitializerBuilder":
        """
        Add multiple InitStep objects at once.

        Args:
            initializers: List of InitStep objects

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
            if isinstance(m, BuildsInitSteps):
                self.extend(m.run_build())
        return self

    def build(self) -> List[InitStep]:
        """
        Return the constructed list of initializers.

        A step whose name is listed in another step's ``replaces=`` is dropped
        so the replacer can supersede a default step. A replacer keeps its own
        name in its ``replaces=`` (the terminal-alias pattern), so it is never
        dropped by its own declaration and dependents still resolve.

        Returns:
            List of InitStep objects
        """
        replaced = {name for step in self.initializers for name in step.replaces}
        return [
            step
            for step in self.initializers
            if not (step.name in replaced and step.name not in step.replaces)
        ]
