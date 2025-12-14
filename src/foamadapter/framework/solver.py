# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from typing import Protocol, runtime_checkable

from foamadapter.framework.operations import OperationCollection

from .context import Context
from .decorator import operation
from .initialization import read_files, configure, setup


def Solver(cls: type) -> type:
    """
    A class decorator to mark a class as a Solver in the framework.
    Solvers define the main simulation loop and the basic execution of operations.
    Can be extended via Models.
    """
    return cls


Solver.operation = staticmethod(operation)  # type: ignore[attr-defined]
Solver.read_files = staticmethod(read_files)  # type: ignore[attr-defined]
Solver.configure = staticmethod(configure)  # type: ignore[attr-defined]
Solver.setup = staticmethod(setup)  # type: ignore[attr-defined]


@runtime_checkable
class SolverInterface(Protocol):
    def operations(self, domain_name: str | None = None) -> OperationCollection: ...

    def main_loop(self, ctx: Context) -> None: ...
