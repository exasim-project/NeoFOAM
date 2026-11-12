# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""heatConductionNeoN steps the postProcess model: a declared table becomes a CSV.

``cases/bar`` with ``cases/barTables`` laid over it: the mean of ``T`` at every
one of the 50 steps. The bar's cells are equal, so the last row is the mean of
the final ``T`` (``test_bar_decay.DISCRETE_T``), up to the CSV's digits. The
graph test pins the place of the postProcess operations, last in the time loop,
with the core models faked down to their spec name and op names.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neofoam.framework.context import Context
from neofoam.framework.model import ModelRuntime
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata
from neofoam.solver.heatConductionNeoN.heatConductionNeoN import execution_graph
from solver.heatConductionNeoN._run_case import mesh_copy, result
from solver.heatConductionNeoN.test_bar_decay import DISCRETE_T


def test_a_declared_table_gets_the_mean_T_of_every_step(tmp_path: Path) -> None:
    case_dir = mesh_copy("bar", tmp_path / "bar", "barTables")
    result(case_dir, "single")

    with (case_dir / "postProcessing" / "mean_T.csv").open(newline="") as handle:
        header, *rows = csv.reader(handle)

    assert header == ["time", "mean_T"]
    assert len(rows) == 50
    assert float(rows[-1][0]) == pytest.approx(0.05)
    assert float(rows[-1][1]) == pytest.approx(np.mean(DISCRETE_T), rel=0, abs=1e-6)


def _core_model(spec_name: str, op_names: list[str]) -> ModelRuntime:
    operations = [
        Operation(func=SequentialOp(lambda _ctx: None), metadata=OperationMetadata(op_name=name))
        for name in op_names
    ]
    return ModelRuntime(
        spec=SimpleNamespace(name=spec_name, _build_operations_for=lambda _rt: operations),
        name=spec_name,
        config=None,
    )


def test_the_time_loop_ends_with_write_output_then_the_post_process_operations() -> None:
    state = SimpleNamespace(
        core_models=[
            _core_model("solutionLoop", ["set_time_step", "increment_time"]),
            _core_model("fieldWriter", ["write_output"]),
            _core_model("postProcess", ["post_process", "insitu"]),
        ]
    )
    solver = SimpleNamespace(
        state=state,
        operations={
            "start_step": _core_model("x", ["start_step"]).operations[0],
            "solve_T": _core_model("x", ["solve_T"]).operations[0],
        },
    )

    builder, _ = execution_graph(solver, Context(fields={}, models={}))

    time_loop = builder.operations["time_loop"]
    assert [op.operation_name for op in time_loop.sub_operations][-3:] == [
        "write_output",
        "post_process",
        "insitu",
    ]
