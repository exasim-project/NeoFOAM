# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for DAGResolver — tests each pure function in isolation,
plus integration tests through the ``DAGResolver.resolve()`` orchestrator."""

from __future__ import annotations

from typing import Any

import pytest

from neofoam.framework.context import Context
from neofoam.framework.graph import (
    CyclicDependencyError,
    DAGResolver,
    MissingDependencyError,
    NetworkxTopologicalSorter,
)
from neofoam.framework.graph.resolver import (
    _build_global_graph as build_global_graph,
    _collect_tagged_ops as collect_tagged_ops,
    _infer_target_scope as infer_target_scope,
    _rebuild_builder as rebuild_builder,
)
from neofoam.framework.graph.resolver import _sort_global as _sort_global_impl
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    SequentialOp,
    StepBuilder,
)
from neofoam.framework.types import OperationMetadata, OperationNumber

from framework.conftest import MaxIterations


def sort_global(graph, op_map, tagged, sorter=None):
    """Test-local wrapper that defaults the sorter for brevity in assertions."""
    return _sort_global_impl(
        graph, op_map, tagged, sorter=sorter or NetworkxTopologicalSorter()
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop(ctx: Any) -> None:
    """No-op callable for structural tests."""
    pass


def _seq(
    name: str,
    number: str | None = None,
    depends_on: list[str] | None = None,
    before: list[str] | None = None,
) -> Operation:
    """Create a sequential Operation with the given metadata."""
    return Operation(
        func=SequentialOp(_noop),
        metadata=OperationMetadata(
            op_name=name,
            operation_number=OperationNumber(number) if number is not None else None,
            depends_on=depends_on,
            before=before,
        ),
    )


def _loop_op(name: str, number: str | None = None, max_iters: int = 1) -> Operation:
    """Create an iterative (loop) Operation."""
    return Operation(
        func=IterativeOp(MaxIterations(max_iters=max_iters)),
        metadata=OperationMetadata(
            op_name=name,
            operation_number=OperationNumber(number) if number is not None else None,
        ),
    )


def _names_in(builder: StepBuilder, path: list[int] | None = None) -> list[str | None]:
    """Extract operation names from a resolved builder.

    path: list of indices to descend into sub_operations.
          e.g. [0, 0] = first op's first sub_op's sub_operations.
    """
    ops = builder.operations.ops
    for idx in path or []:
        ops = ops[idx].sub_operations
    return [op.operation_name for op in ops]


def _build_single_loop() -> StepBuilder:
    """Builder with root → time_loop → [step1, step2, step3]."""
    builder = StepBuilder()
    loop = _loop_op("time_loop", number="1")
    with builder.loop(loop) as lb:
        lb.step(_seq("step1", number="1"))
        lb.step(_seq("step2", number="2", depends_on=["step1"]))
        lb.step(_seq("step3", number="3", depends_on=["step2"]))
    return builder


def _build_nested() -> StepBuilder:
    """Builder with root → time_loop → inner_loop → [step1, step2, step3]."""
    builder = StepBuilder()
    time_loop = _loop_op("time_loop", number="1")
    with builder.loop(time_loop) as tl:
        inner_loop = _loop_op("inner_loop", number="1")
        with tl.loop(inner_loop) as il:
            il.step(_seq("step1", number="1"))
            il.step(_seq("step2", number="2", depends_on=["step1"]))
            il.step(_seq("step3", number="3", depends_on=["step2"]))
    return builder


# ===========================================================================
# 1. collect_tagged_ops
# ===========================================================================


def test_collect_flat_builder() -> None:
    """Flat builder tags all ops with scope 'root'."""
    builder = StepBuilder()
    builder.step(_seq("A", number="1"))
    builder.step(_seq("B", number="2"))

    tagged = collect_tagged_ops(builder, Operations())
    scopes = {name: scope for scope, op in tagged if (name := op.operation_name)}
    assert scopes == {"A": "root", "B": "root"}


@pytest.mark.parametrize(
    "builder, model_ops, expected",
    [
        (
            _build_single_loop(),
            Operations(),
            {
                "time_loop": "root",
                "step1": "time_loop",
                "step2": "time_loop",
                "step3": "time_loop",
            },
        ),
        (
            _build_nested(),
            Operations(),
            {
                "time_loop": "root",
                "inner_loop": "time_loop",
                "step1": "inner_loop",
                "step2": "inner_loop",
                "step3": "inner_loop",
            },
        ),
        (
            _build_single_loop(),
            Operations([_seq("M1", number="1.5", depends_on=["step1"])]),
            {"M1": "time_loop"},
        ),
        (
            _build_single_loop(),
            Operations(
                [
                    _seq("M1", number="1.5", depends_on=["step1"]),
                    _seq("M2", number="1.7", depends_on=["M1"]),
                ]
            ),
            {"M1": "time_loop", "M2": "time_loop"},
        ),
        (
            _build_nested(),
            Operations([_seq("M_free", number="0.5")]),
            {"M_free": "inner_loop"},
        ),
    ],
    ids=[
        "single_loop_scopes",
        "nested_loop_scopes",
        "model_ops_with_deps_placed_in_loop",
        "model_ops_chained",
        "model_ops_no_deps_default_to_innermost",
    ],
)
def test_collect_scopes(builder, model_ops, expected) -> None:
    """Loop/model ops are tagged with the expected scope at each nesting level.

    Each case asserts the scope of the named ops it cares about (the builder may
    contain more ops than the case checks).
    """
    tagged = collect_tagged_ops(builder, model_ops)
    scopes = {op.operation_name: scope for scope, op in tagged}
    for name, scope in expected.items():
        assert scopes[name] == scope


def test_collect_empty_builder_no_model_ops() -> None:
    """Empty builder + empty model ops → empty list."""
    builder = StepBuilder()
    tagged = collect_tagged_ops(builder, Operations())
    assert tagged == []


# ===========================================================================
# 2. infer_target_scope
# ===========================================================================


@pytest.mark.parametrize(
    "op, kwargs, expected",
    [
        (
            _seq("M1", depends_on=["step1"]),
            dict(op_to_scope={"step1": "time_loop"}, all_scopes={"root", "time_loop"}),
            "time_loop",
        ),
        (
            _seq("M1", before=["step2"]),
            dict(
                op_to_scope={"step2": "inner_loop"},
                all_scopes={"root", "time_loop", "inner_loop"},
            ),
            "inner_loop",
        ),
        (
            _seq("M1", depends_on=["root_op", "loop_op"]),
            dict(
                op_to_scope={"root_op": "root", "loop_op": "time_loop"},
                all_scopes={"root", "time_loop"},
            ),
            "time_loop",
        ),
        (
            _seq("M1", depends_on=["A"]),
            dict(op_to_scope={"A": "root"}, all_scopes={"root", "time_loop"}),
            "root",
        ),
        (
            _seq("M1"),
            dict(
                op_to_scope={},
                all_scopes={"root", "inner_loop", "time_loop"},
                scope_depth={"root": 0, "time_loop": 1, "inner_loop": 2},
            ),
            "inner_loop",
        ),
        (
            _seq("M1"),
            dict(
                op_to_scope={},
                all_scopes={"root", "a_outer", "z_inner"},
                scope_depth={"root": 0, "a_outer": 1, "z_inner": 2},
            ),
            "z_inner",
        ),
        (
            _seq("M1"),
            dict(op_to_scope={}, all_scopes={"root"}),
            "root",
        ),
        (
            _seq("M1", depends_on=["unknown_op"]),
            dict(op_to_scope={"step1": "time_loop"}, all_scopes={"root", "time_loop"}),
            "time_loop",
        ),
    ],
    ids=[
        "from_depends_on",
        "from_before",
        "prefers_non_root",
        "all_root_returns_root",
        "no_deps_falls_back_to_deepest",
        "no_deps_deepest_adversarial_names",
        "no_deps_no_loops_returns_root",
        "unknown_dep_ignored",
    ],
)
def test_infer_target_scope(op, kwargs, expected) -> None:
    """Scope is inferred from deps/before targets, preferring deeper loop scopes,
    falling back to the deepest scope by nesting depth when there are no deps."""
    assert infer_target_scope(op, **kwargs) == expected


# ===========================================================================
# 3. build_global_graph
# ===========================================================================


def test_graph_flat_ops_become_nodes() -> None:
    """Sequential ops become graph nodes; their names are used."""
    tagged = [
        ("root", _seq("A", number="1")),
        ("root", _seq("B", number="2")),
    ]
    graph, op_map = build_global_graph(tagged)

    assert set(graph.nodes) == {"A", "B"}
    assert len(graph.edges) == 0
    assert "A" in op_map and "B" in op_map


def test_graph_loop_ops_excluded_from_nodes() -> None:
    """Loop (IterativeOp) ops are not graph nodes."""
    tagged = [
        ("root", _loop_op("time_loop")),
        ("time_loop", _seq("step1", number="1")),
    ]
    graph, op_map = build_global_graph(tagged)

    assert "time_loop" not in graph.nodes
    assert "step1" in graph.nodes


def test_graph_depends_on_creates_edge() -> None:
    """depends_on creates a directed edge dep → dependent."""
    tagged = [
        ("root", _seq("A", number="1")),
        ("root", _seq("B", number="2", depends_on=["A"])),
    ]
    graph, _ = build_global_graph(tagged)

    assert graph.has_edge("A", "B")
    assert not graph.has_edge("B", "A")


def test_graph_before_creates_edge() -> None:
    """before creates a directed edge op → before_target."""
    tagged = [
        ("root", _seq("X", number="1", before=["Y"])),
        ("root", _seq("Y", number="2")),
    ]
    graph, _ = build_global_graph(tagged)

    assert graph.has_edge("X", "Y")


def test_graph_diamond_edges() -> None:
    """Diamond: A→B, A→C, B→D, C→D creates correct edges."""
    tagged = [
        ("root", _seq("A", number="1")),
        ("root", _seq("B", number="2", depends_on=["A"])),
        ("root", _seq("C", number="3", depends_on=["A"])),
        ("root", _seq("D", number="4", depends_on=["B", "C"])),
    ]
    graph, _ = build_global_graph(tagged)

    assert graph.has_edge("A", "B")
    assert graph.has_edge("A", "C")
    assert graph.has_edge("B", "D")
    assert graph.has_edge("C", "D")
    assert len(graph.edges) == 4


@pytest.mark.parametrize(
    "tagged",
    [
        [("root", _seq("A", depends_on=["ghost"]))],
        [("root", _seq("A", before=["ghost"]))],
    ],
    ids=["missing_depends_on", "missing_before_target"],
)
def test_graph_missing_target_raises(tagged) -> None:
    """depends_on / before targeting a non-existent op raises MissingDependencyError."""
    with pytest.raises(MissingDependencyError, match="ghost"):
        build_global_graph(tagged)


def test_graph_cross_scope_edges() -> None:
    """Ops in different scopes still create edges in the global graph."""
    tagged = [
        ("time_loop", _seq("step1", number="1")),
        ("inner_loop", _seq("step2", number="2", depends_on=["step1"])),
    ]
    graph, _ = build_global_graph(tagged)

    assert graph.has_edge("step1", "step2")


def test_graph_scope_stored_as_node_attribute() -> None:
    """Scope is stored as node attribute on the graph."""
    tagged = [
        ("time_loop", _seq("step1", number="1")),
        ("inner_loop", _seq("step2", number="2")),
    ]
    graph, _ = build_global_graph(tagged)

    assert graph.nodes["step1"]["scope"] == "time_loop"
    assert graph.nodes["step2"]["scope"] == "inner_loop"


# ===========================================================================
# 4. sort_global
# ===========================================================================


@pytest.mark.parametrize(
    "tagged, expected",
    [
        (
            [
                ("root", _seq("C", number="3")),
                ("root", _seq("A", number="1")),
                ("root", _seq("B", number="2")),
            ],
            ["A", "B", "C"],
        ),
        (
            [
                ("root", _seq("C", number="3", depends_on=["B"])),
                ("root", _seq("A", number="1")),
                ("root", _seq("B", number="2", depends_on=["A"])),
            ],
            ["A", "B", "C"],
        ),
        (
            [
                ("root", _seq("op_2", number="2")),
                ("root", _seq("op_1_1", number="1.1")),
                ("root", _seq("op_1", number="1")),
                ("root", _seq("op_1_0_1", number="1.0.1")),
            ],
            ["op_1", "op_1_0_1", "op_1_1", "op_2"],
        ),
    ],
    ids=[
        "independent_by_operation_number",
        "chain_dependency",
        "sub_version_numbers",
    ],
)
def test_sort_root_exact_order(tagged, expected) -> None:
    """Root-scope ops sort into the exact expected order (number + dependency edges)."""
    graph, op_map = build_global_graph(tagged)
    result = sort_global(graph, op_map, tagged)

    names = [op.operation_name for op in result["root"]]
    assert names == expected


@pytest.mark.parametrize(
    "tagged, earlier, later",
    [
        (
            [
                ("root", _seq("A", number="1", depends_on=["B"])),
                ("root", _seq("B", number="5")),
            ],
            "B",
            "A",
        ),
        (
            [
                ("root", _seq("numbered", number="1")),
                ("root", _seq("unnumbered", number=None)),
            ],
            "numbered",
            "unnumbered",
        ),
    ],
    ids=["dependency_overrides_number", "unnumbered_after_numbered"],
)
def test_sort_root_relative_order(tagged, earlier, later) -> None:
    """*earlier* sorts before *later* (dependency edges and number tie-breaking)."""
    graph, op_map = build_global_graph(tagged)
    result = sort_global(graph, op_map, tagged)

    names = [op.operation_name for op in result["root"]]
    assert names.index(earlier) < names.index(later)


def test_sort_groups_by_scope() -> None:
    """Ops grouped by their scope in the output dict."""
    tagged = [
        ("root", _loop_op("time_loop")),
        ("time_loop", _seq("s1", number="1")),
        ("time_loop", _seq("s2", number="2")),
        ("root", _seq("r1", number="1")),
    ]
    graph, op_map = build_global_graph(tagged)
    result = sort_global(graph, op_map, tagged)

    assert "root" in result
    assert "time_loop" in result
    time_loop_names = [op.operation_name for op in result["time_loop"]]
    assert time_loop_names == ["s1", "s2"]


def test_sort_loop_ops_injected_into_parent_scope() -> None:
    """Loop (IterativeOp) ops are re-injected into their parent scope."""
    tagged = [
        ("root", _loop_op("time_loop")),
        ("time_loop", _seq("step1", number="1")),
    ]
    graph, op_map = build_global_graph(tagged)
    result = sort_global(graph, op_map, tagged)

    root_names = [op.operation_name for op in result["root"]]
    assert "time_loop" in root_names


def test_sort_cyclic_raises() -> None:
    """Cyclic dependency raises CyclicDependencyError."""
    tagged = [
        ("root", _seq("A", number="1", depends_on=["B"])),
        ("root", _seq("B", number="2", depends_on=["A"])),
    ]
    graph, op_map = build_global_graph(tagged)

    with pytest.raises(CyclicDependencyError):
        sort_global(graph, op_map, tagged)


# ===========================================================================
# 5. rebuild_builder
# ===========================================================================


def test_rebuild_flat_from_sorted_scopes() -> None:
    """Flat builder reconstructed from sorted_scopes['root']."""
    original = StepBuilder()
    original.step(_seq("B", number="2"))
    original.step(_seq("A", number="1"))

    sorted_scopes = {
        "root": [_seq("A", number="1"), _seq("B", number="2")],
    }

    result = rebuild_builder(original, sorted_scopes)
    assert _names_in(result) == ["A", "B"]


def test_rebuild_single_loop() -> None:
    """Single-loop builder reconstructed with sorted loop body."""
    original = _build_single_loop()

    sorted_scopes = {
        "root": [_loop_op("time_loop")],
        "time_loop": [
            _seq("step1", number="1"),
            _seq("step2", number="2"),
            _seq("step3", number="3"),
        ],
    }

    result = rebuild_builder(original, sorted_scopes)
    assert result.operations[0].operation_name == "time_loop"
    inner = _names_in(result, path=[0])
    assert inner == ["step1", "step2", "step3"]


def test_rebuild_nested_loops() -> None:
    """Nested loops reconstructed correctly from sorted_scopes."""
    original = _build_nested()

    sorted_scopes = {
        "root": [_loop_op("time_loop")],
        "time_loop": [_loop_op("inner_loop")],
        "inner_loop": [
            _seq("step1", number="1"),
            _seq("step2", number="2"),
            _seq("step3", number="3"),
        ],
    }

    result = rebuild_builder(original, sorted_scopes)
    assert result.operations[0].operation_name == "time_loop"
    tl_ops = result.operations[0].sub_operations
    assert tl_ops[0].operation_name == "inner_loop"
    inner = _names_in(result, path=[0, 0])
    assert inner == ["step1", "step2", "step3"]


def test_rebuild_empty_builder() -> None:
    """Empty builder + empty sorted_scopes → empty result."""
    original = StepBuilder()
    result = rebuild_builder(original, {})
    assert len(result.operations) == 0


def test_rebuild_preserves_loop_op_identity() -> None:
    """Rebuilt loop op has same name as original."""
    original = _build_single_loop()

    sorted_scopes = {
        "root": [_loop_op("time_loop")],
        "time_loop": [_seq("step1", number="1")],
    }

    result = rebuild_builder(original, sorted_scopes)
    loop = result.operations[0]
    assert loop.operation_name == "time_loop"
    assert isinstance(loop.func, IterativeOp)


def test_rebuild_with_extra_model_ops() -> None:
    """Sorted scopes contain model ops not in the original builder."""
    original = _build_single_loop()

    sorted_scopes = {
        "root": [_loop_op("time_loop")],
        "time_loop": [
            _seq("step1", number="1"),
            _seq("M1", number="1.5"),
            _seq("step2", number="2"),
            _seq("step3", number="3"),
        ],
    }

    result = rebuild_builder(original, sorted_scopes)
    inner = _names_in(result, path=[0])
    assert inner == ["step1", "M1", "step2", "step3"]


# ===========================================================================
# 6. Integration tests — DAGResolver.resolve()
# ===========================================================================


def test_resolve_flat_no_ops() -> None:
    """Empty builder + empty model ops → empty result."""
    builder = StepBuilder()
    result = DAGResolver().resolve(builder, Operations())
    assert len(result.operations) == 0


def test_resolve_flat_single_op() -> None:
    builder = StepBuilder()
    builder.step(_seq("A", number="1"))
    result = DAGResolver().resolve(builder, Operations())
    assert _names_in(result) == ["A"]


def test_resolve_flat_sorted_by_number() -> None:
    """Independent ops ordered by OperationNumber."""
    builder = StepBuilder()
    builder.step(_seq("C", number="3"))
    builder.step(_seq("A", number="1"))
    builder.step(_seq("B", number="2"))

    result = DAGResolver().resolve(builder, Operations())
    assert _names_in(result) == ["A", "B", "C"]


def test_resolve_flat_deps_override_number() -> None:
    """Dependencies override OperationNumber ordering."""
    builder = StepBuilder()
    builder.step(_seq("A", number="1", depends_on=["B"]))
    builder.step(_seq("B", number="2"))

    result = DAGResolver().resolve(builder, Operations())
    names = _names_in(result)
    assert names.index("B") < names.index("A")


def test_resolve_flat_before_constraint() -> None:
    builder = StepBuilder()
    builder.step(_seq("X", number="3", before=["Y"]))
    builder.step(_seq("Y", number="1"))

    result = DAGResolver().resolve(builder, Operations())
    names = _names_in(result)
    assert names.index("X") < names.index("Y")


def test_resolve_flat_diamond() -> None:
    builder = StepBuilder()
    builder.step(_seq("D", number="4", depends_on=["B", "C"]))
    builder.step(_seq("B", number="2", depends_on=["A"]))
    builder.step(_seq("C", number="3", depends_on=["A"]))
    builder.step(_seq("A", number="1"))

    result = DAGResolver().resolve(builder, Operations())
    names = _names_in(result)
    assert names[0] == "A"
    assert names[-1] == "D"


def test_resolve_flat_model_ops_in_root() -> None:
    builder = StepBuilder()
    builder.step(_seq("S1", number="1"))
    model_ops = Operations([_seq("M1", number="1.5", depends_on=["S1"])])

    result = DAGResolver().resolve(builder, model_ops)
    names = _names_in(result)
    assert names.index("S1") < names.index("M1")


def test_resolve_single_loop_structure() -> None:
    builder = _build_single_loop()
    result = DAGResolver().resolve(builder, Operations())

    assert result.operations[0].operation_name == "time_loop"
    assert _names_in(result, path=[0]) == ["step1", "step2", "step3"]


def test_resolve_single_loop_model_ops() -> None:
    builder = _build_single_loop()
    model_ops = Operations(
        [
            _seq("M1", number="1.5", depends_on=["step1"]),
        ]
    )

    result = DAGResolver().resolve(builder, model_ops)
    inner = _names_in(result, path=[0])
    assert inner.index("step1") < inner.index("M1")


def test_resolve_nested_structure() -> None:
    builder = _build_nested()
    result = DAGResolver().resolve(builder, Operations())

    assert result.operations[0].operation_name == "time_loop"
    assert result.operations[0].sub_operations[0].operation_name == "inner_loop"
    assert _names_in(result, path=[0, 0]) == ["step1", "step2", "step3"]


def test_resolve_nested_model_ops_in_inner() -> None:
    builder = _build_nested()
    model_ops = Operations(
        [
            _seq("M1", number="2.5", depends_on=["step2"]),
        ]
    )

    result = DAGResolver().resolve(builder, model_ops)
    inner = _names_in(result, path=[0, 0])
    assert "M1" in inner
    assert inner.index("step2") < inner.index("M1")


def test_resolve_nested_chained_model_ops() -> None:
    builder = _build_nested()
    model_ops = Operations(
        [
            _seq("M_first", number="1.5", depends_on=["step1"]),
            _seq("M_second", number="1.7", depends_on=["M_first"]),
        ]
    )

    result = DAGResolver().resolve(builder, model_ops)
    inner = _names_in(result, path=[0, 0])
    assert inner.index("step1") < inner.index("M_first")
    assert inner.index("M_first") < inner.index("M_second")


def test_resolve_cyclic_raises() -> None:
    builder = StepBuilder()
    builder.step(_seq("A", number="1", depends_on=["B"]))
    builder.step(_seq("B", number="2", depends_on=["A"]))

    with pytest.raises(CyclicDependencyError):
        DAGResolver().resolve(builder, Operations())


def test_resolve_missing_dep_raises() -> None:
    builder = StepBuilder()
    builder.step(_seq("A", number="1", depends_on=["nonexistent"]))

    with pytest.raises(MissingDependencyError, match="nonexistent"):
        DAGResolver().resolve(builder, Operations())


def test_resolve_realistic_two_models() -> None:
    """Full solver with two models through the complete pipeline."""
    builder = _build_nested()
    model_ops = Operations(
        [
            _seq("m1_s1", number="2.5", depends_on=["step1"]),
            _seq("m1_s2", number="2.7", depends_on=["m1_s1"]),
            _seq("m2_s1", number="2.8", depends_on=["step2"]),
        ]
    )

    result = DAGResolver().resolve(builder, model_ops)
    inner = _names_in(result, path=[0, 0])
    assert inner == ["step1", "step2", "m1_s1", "m1_s2", "m2_s1", "step3"]


# ===========================================================================
# 7. Runnable operations (verify execution still works after resolve)
# ===========================================================================


def test_runnable_flat_execution_order() -> None:
    """Run resolved flat operations and check execution order."""
    log: list[str] = []

    def make_logger(name: str) -> Any:
        def fn(ctx: Context) -> None:
            log.append(name)

        return fn

    builder = StepBuilder()
    builder.step(
        Operation(
            func=SequentialOp(make_logger("B")),
            metadata=OperationMetadata(
                op_name="B", operation_number=OperationNumber("2")
            ),
        )
    )
    builder.step(
        Operation(
            func=SequentialOp(make_logger("A")),
            metadata=OperationMetadata(
                op_name="A", operation_number=OperationNumber("1")
            ),
        )
    )

    result = DAGResolver().resolve(builder, Operations())
    result.operations.run(Context(fields={}, models={}, mesh={}))

    assert log == ["A", "B"]


def test_runnable_loop_execution_order_with_model_ops() -> None:
    """Run nested loop with model ops and verify execution sequence."""
    log: list[str] = []

    def make_logger(name: str) -> Any:
        def fn(ctx: Context) -> None:
            log.append(name)

        return fn

    builder = StepBuilder()
    loop = Operation(
        func=IterativeOp(MaxIterations(max_iters=2)),
        metadata=OperationMetadata(op_name="loop"),
    )
    with builder.loop(loop) as lb:
        lb.step(
            Operation(
                func=SequentialOp(make_logger("S1")),
                metadata=OperationMetadata(
                    op_name="S1", operation_number=OperationNumber("1")
                ),
            )
        )
        lb.step(
            Operation(
                func=SequentialOp(make_logger("S2")),
                metadata=OperationMetadata(
                    op_name="S2",
                    operation_number=OperationNumber("2"),
                    depends_on=["S1"],
                ),
            )
        )

    model_ops = Operations(
        [
            Operation(
                func=SequentialOp(make_logger("M1")),
                metadata=OperationMetadata(
                    op_name="M1",
                    operation_number=OperationNumber("1.5"),
                    depends_on=["S1"],
                ),
            )
        ]
    )

    result = DAGResolver().resolve(builder, model_ops)
    result.operations.run(Context(fields={}, models={}, mesh={}))

    # 2 iterations × (S1, M1, S2) = [S1, M1, S2, S1, M1, S2]
    assert log == ["S1", "M1", "S2", "S1", "M1", "S2"]
