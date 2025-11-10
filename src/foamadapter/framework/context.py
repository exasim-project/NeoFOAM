from typing import Annotated, Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
Model = Annotated[T, "models"]
Field = Annotated[T, "fields"]


class FieldUpdates(dict):
    """
        A dictionary that holds updates to fields in the Context object.
    """

    pass


class Context(BaseModel):
    """
        The Context object holds the state of the framework at a given point in time.
        It contains fields and models that are used by various components of the framework.

        The relevant fields or models are injected into the operations 
    """
    model_config = {"arbitrary_types_allowed": True}
    fields: dict[str, Any]
    models: dict[str, Any]
    mesh: Any = None
    runTime: Any = None

