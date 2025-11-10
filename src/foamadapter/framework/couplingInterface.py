from pydantic import BaseModel


class CouplingInterface(BaseModel):
    type: str
