from typing import Literal, Optional

from pydantic import BaseModel

from neofoam.framework.context import Context
from neofoam.framework.operations import OperationCollection
from neofoam.framework.simulation import Domain
from neofoam.framework.solver import Solver


@Solver
class MyCustomSolver(BaseModel):
    name: Literal["MyCustomSolver"] = "MyCustomSolver"

    def operations(self, domain_name: Optional[str] = None) -> OperationCollection:
        return OperationCollection()

    def main_loop(self, ctx: Context) -> None:
        pass


def test_domain():
    domain = Domain(name="test_domain", solver=MyCustomSolver())
    assert domain.name == "test_domain"
    assert isinstance(domain.solver, MyCustomSolver)
