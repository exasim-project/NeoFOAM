# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A model's operations carry the model as their domain and their docstring summary."""

from typing import Any

from neofoam.framework.graph import DAGResolver
from neofoam.framework.model import ModelRuntime, ModelSpec
from neofoam.framework.operations import Operation, Operations, SequentialOp, StepBuilder
from neofoam.framework.types import OperationMetadata


def _spec() -> ModelSpec:
    spec = ModelSpec("heat")

    @spec.operation(name="solve")
    def solve(self: Any, ctx: Any) -> None:
        """Solve the energy equation.

        Details that are not part of the summary.
        """

    @spec.operation(name="report", depends_on=["solve"])
    def report(self: Any, ctx: Any) -> None:
        pass

    return spec


def _runtime(spec: ModelSpec) -> ModelRuntime:
    return ModelRuntime(spec=spec, name=spec.name, config={})


def test_operations_get_model_domain_and_doc_summary() -> None:
    ops = _runtime(_spec()).operations
    assert [op.metadata.domain_name for op in ops] == ["heat", "heat"]
    assert ops[0].metadata.description == "Solve the energy equation."
    assert ops[1].metadata.description == ""


def test_explicit_metadata_is_kept() -> None:
    spec = ModelSpec("algo")

    @spec.operation_collection
    def collect(self: Any) -> Operations:
        return Operations(
            [
                Operation(
                    func=SequentialOp(lambda ctx: None),
                    metadata=OperationMetadata(
                        op_name="own", domain_name="custom", description="Kept."
                    ),
                ),
                Operation(
                    func=SequentialOp(lambda ctx: None), metadata=OperationMetadata(op_name="bare")
                ),
            ]
        )

    ops = _runtime(spec).operations
    assert [(op.metadata.domain_name, op.metadata.description) for op in ops] == [
        ("custom", "Kept."),
        ("algo", ""),
    ]


def test_domain_does_not_change_resolution() -> None:
    ops = _runtime(_spec()).operations
    resolved = DAGResolver().resolve(StepBuilder(), Operations(list(reversed(ops))))
    assert [op.operation_name for op in resolved.operations.ops] == ["solve", "report"]
