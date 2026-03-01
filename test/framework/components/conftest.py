# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Shared test stubs and factories for framework component tests."""

from typing import Any

from neofoam.io import BaseConfig
from neofoam.framework.operations import Operation, SequentialOp, IterativeOp
from neofoam.framework.types import OperationMetadata, OperationNumber

from framework.conftest import MaxIterations


# ---------------------------------------------------------------------------
# Shared BaseConfig stubs
# ---------------------------------------------------------------------------


class MyConfig(BaseConfig):
    x: int = 1


class OtherConfig(BaseConfig):
    y: int = 2


class StepConfig(BaseConfig):
    factor: float = 0.01


class StubConfig(BaseConfig):
    factor: float = 2.0


# ---------------------------------------------------------------------------
# Shared operation factories
# ---------------------------------------------------------------------------


def noop(ctx: Any) -> None:
    """No-op callable for structural tests."""
    pass


def make_seq_op(
    name: str,
    number: str | int | None = None,
    depends_on: list[str] | None = None,
    before: list[str] | None = None,
    func: Any = None,
    **kw: Any,
) -> Operation:
    """Create a sequential Operation with the given metadata."""
    return Operation(
        func=SequentialOp(func or noop),
        metadata=OperationMetadata(
            op_name=name,
            operation_number=OperationNumber(number) if number is not None else None,
            depends_on=depends_on,
            before=before,
            **kw,
        ),
    )


def make_loop_op(
    name: str,
    number: str | int | None = None,
    max_iters: int = 1,
) -> Operation:
    """Create an iterative (loop) Operation."""
    return Operation(
        func=IterativeOp(MaxIterations(max_iters=max_iters)),
        metadata=OperationMetadata(
            op_name=name,
            operation_number=OperationNumber(number) if number is not None else None,
        ),
    )
