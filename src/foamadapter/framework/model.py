# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from typing import Protocol, runtime_checkable

from foamadapter.framework.context import Context
from foamadapter.framework.operations import OperationCollection

from .decorator import operation
from .initialization import read_files, configure, setup


def Model(cls: type) -> type:
    """
    A class decorator to mark a class as a Model in the framework.
    Models contain a set of operations that are meant to extend a solver's functionality.
    """
    return cls


Model.operation = staticmethod(operation)  # type: ignore[attr-defined]
Model.read_files = staticmethod(read_files)  # type: ignore[attr-defined]
Model.configure = staticmethod(configure)  # type: ignore[attr-defined]
Model.setup = staticmethod(setup)  # type: ignore[attr-defined]


@runtime_checkable
class ModelInterface(Protocol):
    def operations(self) -> OperationCollection: ...

    def run(self, ctx: Context) -> None: ...
