# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from typing import Protocol, runtime_checkable

from foamadapter.framework.operations import OperationCollection

from .context import Context
from .decorator import step


def Solver(cls: type) -> type:
    return cls


Solver.step = staticmethod(step)  # type: ignore[attr-defined]


@runtime_checkable
class SolverInterface(Protocol):
    def operations(self, domain_name: str | None = None) -> OperationCollection: ...

    def main_loop(self, ctx: Context) -> None: ...
