import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - tests incomplete Protocol implementations"
)

from typing import Literal

from pydantic import BaseModel

from foamadapter.framework.context import Context
from foamadapter.framework.operations import OperationCollection
from foamadapter.framework.simulation import Domain
from foamadapter.framework.bkp_solver import Solver


@Solver
class MyCustomSolver(BaseModel):
    name: Literal["MyCustomSolver"] = "MyCustomSolver"

    def operations(self, domain_name: str | None = None) -> OperationCollection:
        return OperationCollection()

    def main_loop(self, ctx: Context) -> None:
        pass


def test_domain():
    domain = Domain(name="test_domain", solver=MyCustomSolver())
    assert domain.name == "test_domain"
    assert isinstance(domain.solver, MyCustomSolver)
