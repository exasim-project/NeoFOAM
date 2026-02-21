# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from pydantic import BaseModel

from neofoam.framework.operations import OperationCollection

from .solver import SolverInterface
from .types import OperationMetadata


class Domain(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    solver: SolverInterface

    def dependencies(self) -> list[OperationMetadata]:
        ops = self.solver.operations(domain_name=self.name)
        return [op.operation_metadata() for op in ops.ops]

    def operations(self) -> OperationCollection:
        return self.solver.operations(domain_name=self.name)
