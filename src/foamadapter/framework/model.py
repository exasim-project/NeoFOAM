# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from typing import Protocol, runtime_checkable

from foamadapter.framework.context import Context
from foamadapter.framework.operations import OperationCollection

from .decorator import operation


def Model(cls: type) -> type:
    return cls


Model.operation = staticmethod(operation)  # type: ignore[attr-defined]


@runtime_checkable
class ModelInterface(Protocol):
    def operations(self) -> OperationCollection: ...

    def run(self, ctx: Context) -> None: ...
