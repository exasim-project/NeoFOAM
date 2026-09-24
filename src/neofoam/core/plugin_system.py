# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Optional, Type, Union

from pydantic import BaseModel, Field, create_model


@dataclass
class PluginRegistry:
    base_cls: Type[BaseModel]
    discriminator_variable: str
    discriminator: str
    plugin_registry: list[Type[BaseModel]] = field(default_factory=list)
    plugin_model: Optional[Type[BaseModel]] = None

    def get_plugin_names(self) -> list[str]:
        names: list[str] = []
        for plugin_cls in self.plugin_registry:
            model_type = getattr(plugin_cls, "model_type", None)
            names.append(model_type or plugin_cls.__name__)
        return names


class PluginSystem:
    """
    PluginSystem provides a runtime-extensible plugin/config system using
    Pydantic discriminated unions.

    Features:
    - Central registry (_registry) for all plugin base types and their plugin classes.
    - Decorator API for explicit registration of plugin families and plugin config classes.
    - Supports multiple independent plugin families, each with its own registry
      and extensible model, keyed by the base class name — so two base classes
      sharing a name cannot both be families and the second one raises.
    - Uses a PluginRegistry dataclass to store metadata for each plugin base type:
        - base_cls: The plugin base class (usually a Pydantic model).
        - plugin_registry: List of registered plugin config classes for this type.
        - discriminator_variable: Name of the field holding the union (e.g., 'plugin').
        - discriminator: Name of the discriminator field in plugin configs (e.g., 'plugin_type').
        - extensible_model: The dynamically generated Pydantic model for this plugin type.

    Usage:
    1. Decorate your base class:
        @PluginSystem.register(discriminator_variable="plugin", discriminator="plugin_type")
        class PluginBase(BaseModel):
            name: str

    2. Register plugin config classes:
        @PluginBase.register
        class AddOneConfig(BaseModel):
            plugin_type: Literal["add_one"]
            amount: int

    3. Access the extensible model:
        PluginModel = PluginBase._extensible_model

    4. The registry can be queried for all plugin families and their plugins:
    PluginSystem._registry["PluginBase"].plugin_registry

    This design allows runtime extensibility, developer-friendly registration,
    and schema validation for plugin/config systems.
    """

    _registry: dict[str, PluginRegistry] = {}

    @staticmethod
    def register(
        discriminator_variable: str, discriminator: str
    ) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        def base_decorator(base_cls: Type[BaseModel]) -> Type[BaseModel]:
            PluginSystem._reject_foreign_family(base_cls)
            # Store metadata for this base class in the registry as a dataclass
            PluginSystem._registry[base_cls.__name__] = PluginRegistry(
                base_cls=base_cls,
                plugin_registry=[],
                discriminator_variable=discriminator_variable,
                discriminator=discriminator,
                plugin_model=None,
            )

            def plugin_decorator(plugin_cls: Type[BaseModel]) -> Type[BaseModel]:
                registry_obj = PluginSystem._registry[base_cls.__name__]
                registry_obj.plugin_registry.append(plugin_cls)
                registry = registry_obj.plugin_registry
                union: Any = Annotated[
                    Union[tuple(registry)],
                    Field(discriminator=registry_obj.discriminator),
                ]
                model = create_model(  # type: ignore[call-overload]
                    f"{base_cls.__name__}ExtensibleModel",
                    **{registry_obj.discriminator_variable: (union, ...)},
                    __base__=base_cls,
                )
                registry_obj.plugin_model = model
                base_cls.plugin_model = model  # type: ignore[attr-defined]
                return plugin_cls

            base_cls.register = plugin_decorator  # type: ignore[method-assign,assignment]
            # Initial model with no plugins
            registry_obj = PluginSystem._registry[base_cls.__name__]
            union: Any = object
            model = create_model(  # type: ignore[call-overload]
                f"{base_cls.__name__}ExtensibleModel",
                **{registry_obj.discriminator_variable: (union, ...)},
                __base__=base_cls,
            )
            registry_obj.plugin_model = model
            base_cls.plugin_model = model  # type: ignore[attr-defined]

            # Add a classmethod 'create' for user-friendly instantiation
            def create(cls: Type[Any], /, **kwargs: Any) -> Any:
                return cls.plugin_model(**kwargs)

            base_cls.create = classmethod(create)  # type: ignore[attr-defined]
            return base_cls

        return base_decorator

    @staticmethod
    def _reject_foreign_family(base_cls: Type[BaseModel]) -> None:
        """Refuse a family name another class already holds; a re-run for the same class is fine."""
        registered = PluginSystem._registry.get(base_cls.__name__)
        if registered is None:
            return
        previous = registered.base_cls
        # A module reload builds a new class object at the same definition site;
        # that is the same family being declared again, not a clash.
        same_definition_site = (
            previous.__module__ == base_cls.__module__
            and previous.__qualname__ == base_cls.__qualname__
        )
        if previous is base_cls or same_definition_site:
            return
        raise ValueError(
            f"plugin family '{base_cls.__name__}' is already registered by "
            f"{previous.__module__}.{previous.__qualname__}; "
            f"{base_cls.__module__}.{base_cls.__qualname__} cannot register under the same "
            "name because the registry is keyed by class name — rename one of the two base "
            "classes."
        )

    @classmethod
    def get_registered(cls, base_cls_name: str) -> Optional["PluginRegistry"]:
        return cls._registry.get(base_cls_name, None)

    @classmethod
    def remove_plugin_model(cls, base_cls_name: str, registered_class: Type[BaseModel]) -> bool:
        registry = cls._registry.get(base_cls_name, None)
        if registry is None:
            return False
        if registered_class in registry.plugin_registry:
            registry.plugin_registry.remove(registered_class)
            union: Any = Annotated[
                Union[tuple(registry.plugin_registry)],
                Field(discriminator=registry.discriminator),
            ]
            model = create_model(  # type: ignore[call-overload]
                f"{base_cls_name}ExtensibleModel",
                **{registry.discriminator_variable: (union, ...)},
                __base__=registry.base_cls,
            )
            registry.plugin_model = model
            registry.base_cls.plugin_model = model  # type: ignore[attr-defined]
            return True
        return False

    @classmethod
    def list_plugins(cls) -> dict[str, list[Type[BaseModel]]]:
        return {name: reg.plugin_registry for name, reg in cls._registry.items()}
