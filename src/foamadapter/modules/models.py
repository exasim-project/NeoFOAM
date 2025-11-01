from pydantic import BaseModel
import pydantic
from typing import Protocol, Union, Callable, runtime_checkable, Any


@runtime_checkable
class Model(Protocol):
    """Model interface that only requires a description property."""

    @property
    def description(self) -> str:
        ...



@runtime_checkable
class ModelsFactory(Protocol):
    """Callable that creates a Model object."""

    @property
    def dependencies(self) -> list[str]:
        ...

    def __call__(self, deps: dict[str, Any]) -> Any:
        ...


class Models(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    entries: dict[str, Union[Model, ModelsFactory]] = pydantic.Field(default_factory=dict)

    def add_model(self, name: str, model: Union[Model, ModelsFactory]) -> None:
        # Validate that the model implements the correct protocol
        if not isinstance(model, Model) and not isinstance(model, ModelsFactory):
            raise TypeError(f"Expected Model or ModelsFactory for '{name}', got {type(model).__name__}")
        self.entries[name] = model

    def __getitem__(self, name: str) -> Union[Model, ModelsFactory]:
        return self.entries[name]

    def dependencies(self) -> dict[str, set[str]]:
        deps = {}
        for name, item in self.entries.items():
            if callable(item) and hasattr(item, "dependencies"):
                deps[name] = set(item.dependencies)
            else:
                deps[name] = set()
        return deps

    @staticmethod
    def deps(*dependencies: str):
        """Decorator to attach dependencies to a ModelsFactory, setting .dependencies for compatibility."""
        def decorator(factory):
            deps_list = list(dependencies)
            setattr(factory, "dependencies", deps_list)
            return factory
        return decorator


def get_model(deps: dict, key: str) -> Model:
    """Retrieve a Model dependency with type validation."""
    obj = deps[key]
    return obj