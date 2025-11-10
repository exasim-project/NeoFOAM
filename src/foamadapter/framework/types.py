from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering


@total_ordering
class StepNumber:
    def __init__(self, version):
        if isinstance(version, str):
            self.parts = [int(p) for p in version.split(".")]
        elif isinstance(version, (list, tuple)):
            self.parts = list(map(int, version))
        elif isinstance(version, int):
            self.parts = [version]
        else:
            raise TypeError(
                "StepNumber must be initialized with a string, int, or list/tuple of integers"
            )

    def _as_tuple(self, other):
        if not isinstance(other, StepNumber):
            other = StepNumber(other)
        max_len = max(len(self.parts), len(other.parts))
        a = tuple(self.parts + [0] * (max_len - len(self.parts)))
        b = tuple(other.parts + [0] * (max_len - len(other.parts)))
        return a, b

    def __eq__(self, other):
        a, b = self._as_tuple(other)
        return a == b

    def __lt__(self, other):
        a, b = self._as_tuple(other)
        return a < b


class OpType(Enum):
    CONDITION = "condition"
    STEP = "step"


@dataclass
class OperationMetadata:
    """Metadata for operations - describes both decorated functions and DAG nodes."""

    # Core identity
    op_name: str

    # Optional metadata
    op_type: OpType | None = None
    description: str = ""
    step_number: StepNumber | None = None
    depends_on: list[str] | None = None
    domain_name: str | None = None

    # DAG visualization properties
    shape: str = "box"
    color: str | None = None
    used_by: list[str] = field(default_factory=list)

    @property
    def is_step(self) -> bool:
        if self.op_type is None:
            return False
        return self.op_type == OpType.STEP

    @property
    def is_condition(self) -> bool:
        if self.op_type is None:
            return False
        return self.op_type == OpType.CONDITION

    @property
    def name(self) -> str:
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
