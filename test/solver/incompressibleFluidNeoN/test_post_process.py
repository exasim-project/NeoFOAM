# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``postProcess`` seam of incompressibleFluidNeoN — wired, but table-free.

**Case-free.** The NeoN backend cannot be solved here at all (the bindings are a
separate build), and none of the three things this module pins needs a run: the
guard is a function of the loaded ``TableSet``, the schema set is static, and the
loop order is a property of the built graph.

**Why the guard.** Every source hands its pipeline a host numpy array, and the
NeoN bindings still copy host->device only (risk R12) — so a table over NeoN
fields has nothing to read. Rather than let it fail deep inside a pipeline, the
solver refuses the declaration at load. This is stricter than the serial-only
guard of ``incompressibleFluid``: on this backend no table runs at all, so the
parallel case needs no separate branch.

**Loop order.** That ``post_process`` is stepped *after* ``write_output`` is what
a CSV row would mean, and it is asserted on the built graph with the core models
the graph step reads faked down to their spec name and op names.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neofoam.framework.context import Context
from neofoam.framework.model import ModelRuntime
from neofoam.framework.operations import Operation, SequentialOp
from neofoam.framework.types import OperationMetadata
from neofoam.postprocess.table import tables_for_case
from neofoam.solver.incompressibleFluidNeoN import config_classes
from neofoam.solver.incompressibleFluidNeoN.create_fields import _refuse_post_processing_tables
from neofoam.solver.incompressibleFluidNeoN.incompressibleFluidNeoN import execution_graph

_CASES = Path(__file__).parent / "cases"


def test_a_declared_table_is_refused_by_the_neon_backend() -> None:
    tables = tables_for_case(_CASES / "postprocess_yaml")

    with pytest.raises(NotImplementedError, match="volume_p"):
        _refuse_post_processing_tables(tables)


def test_a_case_declaring_no_table_passes_the_guard() -> None:
    tables = tables_for_case(_CASES / "regexSolverKeys")

    assert _refuse_post_processing_tables(tables) is None


def test_post_process_config_is_part_of_the_solver_schema() -> None:
    assert "PostProcessConfig" in {c.__name__ for c in config_classes()}


def _operations(op_names: list[str]) -> list[Operation]:
    return [
        Operation(func=SequentialOp(lambda _ctx: None), metadata=OperationMetadata(op_name=name))
        for name in op_names
    ]


def _core_model(spec_name: str, op_names: list[str]) -> ModelRuntime:
    """A built core model as the graph step reads it: a spec name and named ops."""
    operations = _operations(op_names)
    return ModelRuntime(
        spec=SimpleNamespace(name=spec_name, _build_operations_for=lambda _rt: operations),
        name=spec_name,
        config=None,
    )


def _algorithm_model(op_names: list[str]) -> Any:
    """The pressure-velocity spec, which the graph step uses as its own runtime."""
    operations = _operations(op_names)
    return SimpleNamespace(
        spec=SimpleNamespace(name="pimpleNeoN"),
        _build_operations_for=lambda _self: operations,
    )


def test_the_time_loop_ends_with_write_output_then_post_process() -> None:
    state = SimpleNamespace(
        core_models=[
            _algorithm_model(
                ["rotate_and_report", "inner_loop", "momentum", "continuity", "turbulence_correct"]
            ),
            _core_model("solutionLoop", ["set_time_step", "increment_time"]),
            _core_model("fieldWriter", ["write_output"]),
            _core_model("postProcess", ["post_process"]),
        ],
        optional_models=[],
    )

    builder, _ = execution_graph(SimpleNamespace(state=state), Context(fields={}, models={}))

    time_loop = builder.operations["time_loop"]
    assert [op.operation_name for op in time_loop.sub_operations][-2:] == [
        "write_output",
        "post_process",
    ]
