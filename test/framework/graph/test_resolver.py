# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import pytest

from neofoam.framework.graph import (
    CyclicDependencyError,
    DAGResolver,
    MissingDependencyError,
    NetworkxTopologicalSorter,
)
from neofoam.framework.graph import resolver as resolver_module
from neofoam.framework.graph.resolver import (
    _build_global_graph as build_global_graph,
    _collect_tagged_ops as collect_tagged_ops,
    _infer_target_scope as infer_target_scope,
    _rebuild_builder as rebuild_builder,
    _sort_global as sort_global,
)
from neofoam.framework.operations import (
    IterativeOp,
    Operation,
    Operations,
    StepBuilder,
)
from neofoam.framework.types import OperationNumber


def _seq(name, depends_on=None, before=None, operation_number=None):
    return Operation(
        func=lambda ctx: None,
        operation_name=name,
        depends_on=depends_on,
        before=before,
        operation_number=operation_number,
    )


def _loop(name, sub_operations=None):
    return Operation(
        func=IterativeOp(lambda ctx: False),
        operation_name=name,
        sub_operations=sub_operations or [],
    )


def test_collect_tagged_ops_flat():
    builder = StepBuilder()
    builder.step(_seq("a"))
    builder.step(_seq("b"))

    tagged = collect_tagged_ops(builder, Operations(operations=[]))

    scopes = [scope for scope, _ in tagged]
    names = [op.operation_name for _, op in tagged]
    assert scopes == ["root", "root"]
    assert names == ["a", "b"]


def test_collect_tagged_ops_nested_loop():
    inner = _seq("inner_step")
    loop = _loop("loop1", sub_operations=[inner])
    builder = StepBuilder()
    builder.step(_seq("pre"))
    builder.step(loop)

    tagged = collect_tagged_ops(builder, Operations(operations=[]))

    by_name = {op.operation_name: scope for scope, op in tagged}
    assert by_name["pre"] == "root"
    assert by_name["loop1"] == "root"
    assert by_name["inner_step"] == "loop1"


def test_collect_tagged_ops_model_op_placed_in_dep_scope():
    inside_loop = _seq("inside")
    loop = _loop("loop1", sub_operations=[inside_loop])
    builder = StepBuilder()
    builder.step(loop)

    model_op = _seq("modelOp", depends_on=["inside"])
    tagged = collect_tagged_ops(builder, Operations(operations=[model_op]))

    placed_scope = next(scope for scope, op in tagged if op is model_op)
    assert placed_scope == "loop1"


def test_infer_target_scope_deepest_wins():
    op = _seq("modelOp", depends_on=["shallow_dep", "deep_dep"])
    op_to_scope = {"shallow_dep": "outer", "deep_dep": "inner"}
    all_scopes = {"root", "outer", "inner"}
    scope_depth = {"root": 0, "outer": 1, "inner": 2}

    target = infer_target_scope(op, op_to_scope, all_scopes, scope_depth)

    assert target == "inner"


def test_infer_target_scope_no_deps_fallback():
    op = _seq("modelOp")
    op_to_scope = {}
    all_scopes = {"root", "outer", "inner"}
    scope_depth = {"root": 0, "outer": 1, "inner": 2}

    target = infer_target_scope(op, op_to_scope, all_scopes, scope_depth)

    assert target == "inner"


def test_build_global_graph_edges():
    a = _seq("a")
    b = _seq("b", depends_on=["a"])
    c = _seq("c", before=["b"])
    tagged = [("root", a), ("root", b), ("root", c)]

    graph, op_map = build_global_graph(tagged)

    assert graph.has_edge("a", "b")
    assert graph.has_edge("c", "b")
    assert set(op_map) == {"a", "b", "c"}


def test_build_global_graph_missing_dependency_raises():
    a = _seq("a", depends_on=["does_not_exist"])
    tagged = [("root", a)]

    with pytest.raises(MissingDependencyError):
        build_global_graph(tagged)


def test_build_global_graph_skips_loop_ops():
    loop = _loop("loop1")
    seq = _seq("seq1")
    tagged = [("root", loop), ("root", seq)]

    graph, op_map = build_global_graph(tagged)

    assert "loop1" not in graph.nodes
    assert "loop1" not in op_map
    assert "seq1" in graph.nodes


def test_sort_global_topological():
    a = _seq("a")
    b = _seq("b", depends_on=["a"])
    c = _seq("c", depends_on=["b"])
    tagged = [("root", c), ("root", a), ("root", b)]
    graph, op_map = build_global_graph(tagged)

    sorted_scopes = sort_global(
        graph, op_map, tagged, sorter=NetworkxTopologicalSorter()
    )

    names = [op.operation_name for op in sorted_scopes["root"]]
    assert names == ["a", "b", "c"]


def test_sort_global_operation_number_tiebreak():
    a = _seq("a", operation_number=OperationNumber(2))
    b = _seq("b", operation_number=OperationNumber(1))
    tagged = [("root", a), ("root", b)]
    graph, op_map = build_global_graph(tagged)

    sorted_scopes = sort_global(
        graph, op_map, tagged, sorter=NetworkxTopologicalSorter()
    )

    names = [op.operation_name for op in sorted_scopes["root"]]
    assert names == ["b", "a"]


def test_sort_global_cycle_raises():
    a = _seq("a", depends_on=["b"])
    b = _seq("b", depends_on=["a"])
    tagged = [("root", a), ("root", b)]
    graph, op_map = build_global_graph(tagged)

    with pytest.raises(CyclicDependencyError):
        sort_global(graph, op_map, tagged, sorter=NetworkxTopologicalSorter())


def test_sort_global_loops_reinjected_per_scope():
    loop = _loop("loop1")
    seq = _seq("seq1")
    tagged = [("root", seq), ("root", loop)]
    graph, op_map = build_global_graph(tagged)

    sorted_scopes = sort_global(
        graph, op_map, tagged, sorter=NetworkxTopologicalSorter()
    )

    root_ops = sorted_scopes["root"]
    assert root_ops[0] is loop
    assert root_ops[1] is seq


def test_rebuild_builder_preserves_nesting():
    sub_a = _seq("sub_a")
    sub_b = _seq("sub_b", depends_on=["sub_a"])
    loop = _loop("loop1", sub_operations=[sub_b, sub_a])
    builder = StepBuilder()
    builder.step(loop)

    sorted_scopes = {"root": [loop], "loop1": [sub_a, sub_b]}
    rebuilt = rebuild_builder(builder, sorted_scopes)

    assert len(rebuilt.operations.ops) == 1
    rebuilt_loop = rebuilt.operations.ops[0]
    assert rebuilt_loop.operation_name == "loop1"
    sub_names = [op.operation_name for op in rebuilt_loop.sub_operations]
    assert sub_names == ["sub_a", "sub_b"]


def test_dagresolver_resolve_end_to_end():
    sub_b = _seq("sub_b", depends_on=["sub_a"])
    sub_a = _seq("sub_a")
    loop = _loop("loop1", sub_operations=[sub_b, sub_a])
    builder = StepBuilder()
    builder.step(loop)

    rebuilt = DAGResolver().resolve(builder, Operations(operations=[]))

    rebuilt_loop = rebuilt.operations.ops[0]
    sub_names = [op.operation_name for op in rebuilt_loop.sub_operations]
    assert sub_names == ["sub_a", "sub_b"]


def test_walk_is_module_level():
    assert callable(resolver_module._walk)
    assert resolver_module._walk.__module__ == resolver_module.__name__
