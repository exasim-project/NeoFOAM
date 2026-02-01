# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

from pydantic import BaseModel

from foamadapter.framework.operations import OperationCollection

from .bkp_solver import SolverInterface
from .types import OperationMetadata


class Domain(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    solver: SolverInterface

    def dependencies(self) -> list[OperationMetadata]:
        _, ops = self.solver.execution_graph(domain_name=self.name)
        return [op.operation_metadata() for op in ops.ops]

    def execution_graph(self) -> OperationCollection:
        _, ops = self.solver.execution_graph(domain_name=self.name)
        return ops

    def operations(self) -> OperationCollection:
        _, ops = self.solver.execution_graph(domain_name=self.name)
        return ops
