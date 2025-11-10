from typing import Protocol, runtime_checkable

from foamadapter.framework.operations import OperationCollection

from .context import Context
from .decorator import step


def Solver(cls):
    return cls


Solver.step = staticmethod(step)


@runtime_checkable
class SolverInterface(Protocol):
    def operations(self, domain_name: str | None = None) -> OperationCollection: ...

    def main_loop(self, ctx: Context): ...
