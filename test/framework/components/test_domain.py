from typing import Literal

from pydantic import BaseModel

from foamadapter.framework.context import Context
from foamadapter.framework.simulation import Domain
from foamadapter.framework.solver import Solver


@Solver
class MyCustomSolver(BaseModel):
    name: Literal["MyCustomSolver"] = "MyCustomSolver"

    def operations(self) -> int: ...

    def main_loop(self, ctx: Context): ...


def test_domain():
    domain = Domain(name="test_domain", solver=MyCustomSolver())
    assert domain.name == "test_domain"
    assert isinstance(domain.solver, MyCustomSolver)
