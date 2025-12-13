# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from typing import Protocol, runtime_checkable

from foamadapter.framework.operations import OperationCollection

from .context import Context
from .decorator import operation


def Solver(cls: type) -> type:
    return cls


Solver.operation = staticmethod(operation)  # type: ignore[attr-defined]


@runtime_checkable
class SolverInterface(Protocol):
    def operations(self, domain_name: str | None = None) -> OperationCollection: ...

    def main_loop(self, ctx: Context) -> None: ...
