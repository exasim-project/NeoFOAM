# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from typing import Protocol, Optional, runtime_checkable

from foamadapter.framework.operations import OperationCollection

from .context import Context
from .decorator import operation


def Solver(cls: type) -> type:
    """
    A class decorator to mark a class as a Solver in the framework.
    Solvers define the main simulation loop and the basic execution of operations.
    Can be extended via Models.
    """
    return cls


Solver.operation = operation  # type: ignore[attr-defined]


@runtime_checkable
class SolverInterface(Protocol):
    def operations(self, domain_name: Optional[str] = None) -> OperationCollection: ...

    def main_loop(self, ctx: Context) -> None: ...
