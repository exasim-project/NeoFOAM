# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Core types: OperationMetadata, OpType, OperationNumber."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering


@total_ordering
class OperationNumber:
    """
    The operation number of a solver are typically fixed the operation number allows to easily
    insert new operations between existing ones by incrementing the sub numbers.

    Examples:
        The solver defines the operations with numbers 1, 2, 3.
        Now a new operation needs to be added between 1 and 2, so it is assigned the number 1.1.
        Later another operation is added between 1 and 1.1, which is assigned the number 1.0.1.
    """

    def __init__(self, version: str | int | list[int] | tuple[int, ...]) -> None:
        if isinstance(version, str):
            self.parts = [int(p) for p in version.split(".")]
        elif isinstance(version, (list, tuple)):
            self.parts = list(map(int, version))
        elif isinstance(version, int):
            self.parts = [version]
        else:
            raise TypeError(
                "OperationNumber must be initialized with a string, int, or list/tuple of integers"
            )

    def _as_tuple(
        self, other: OperationNumber | str | int | list[int] | tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not isinstance(other, OperationNumber):
            other = OperationNumber(other)
        max_len = max(len(self.parts), len(other.parts))
        a = tuple(self.parts + [0] * (max_len - len(self.parts)))
        b = tuple(other.parts + [0] * (max_len - len(other.parts)))
        return a, b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (OperationNumber, str, int, list, tuple)):
            return NotImplemented
        a, b = self._as_tuple(other)
        return a == b

    def __lt__(
        self, other: "OperationNumber" | str | int | list[int] | tuple[int, ...]
    ) -> bool:
        a, b = self._as_tuple(other)
        return a < b

    def __str__(self) -> str:
        return ".".join(str(p) for p in self.parts)


class OpType(Enum):
    CONDITION = "condition"
    OPERATION = "operation"


@dataclass
class OperationMetadata:
    """collection of the metadata for operations - describes both decorated functions and DAG nodes."""

    op_name: str | None = None

    # When True the operation is a *fallback* op — scheduled only when a
    # consumer selects the model's fallback backend (e.g. the pybFoam-OpenFOAM
    # turbulence path in incompressibleFluid). Native (non-fallback) ops and
    # fallback ops are partitioned by ``ModelRuntime.native_operations()`` /
    # ``fallback_operations()``; ``.operations`` still returns both.
    fallback: bool = False

    # Optional metadata
    op_type: OpType | None = None
    description: str = ""
    operation_number: OperationNumber | None = None
    depends_on: list[str] | None = None
    before: list[str] | None = None
    domain_name: str | None = None

    # DAG visualization properties
    shape: str = "box"
    color: str | None = None
    used_by: list[str] = field(default_factory=list)

    @property
    def is_operation(self) -> bool:
        if self.op_type is None:
            return False
        return self.op_type == OpType.OPERATION

    @property
    def is_condition(self) -> bool:
        if self.op_type is None:
            return False
        return self.op_type == OpType.CONDITION

    @property
    def name(self) -> str | None:
        if self.op_name is None:
            return None
        if self.domain_name:
            return f"{self.domain_name}.{self.op_name}"
        return self.op_name

    @property
    def dependencies(self) -> list[str]:
        if self.depends_on is None:
            return []
        if self.domain_name:
            return [f"{self.domain_name}.{dep}" for dep in self.depends_on]
        return self.depends_on
