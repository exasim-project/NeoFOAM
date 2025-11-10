from typing import Annotated, Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
Model = Annotated[T, "models"]
Field = Annotated[T, "fields"]


class FieldUpdates(dict):
    """
    A dictionary-like object that signals to the @step decorator
    that its contents should be used to update context.fields.
    """

    pass


class Context(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    fields: dict[str, Any]
    models: dict[str, Any]
    mesh: Any = None
    runTime: Any = None
