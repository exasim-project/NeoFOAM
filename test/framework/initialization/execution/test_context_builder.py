# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for routing init results into a Context via the category router."""

import logging

import pytest

from neofoam.framework.context import Context
from neofoam.framework.initialization.execution import (
    CategoryRouter,
    InitializationGraphError,
    execute_initialization,
)
from neofoam.framework.initialization.execution.context_builder import (
    ContextBuilder,
    build_context_from_results,
    default_router,
)
from neofoam.framework.initialization.execution.init_result import InitResult
from neofoam.framework.initialization.init_step import InitStep


@pytest.mark.parametrize(
    "init_results,expected",
    [
        (
            [
                InitResult("fields.U", "fields", "v"),
                InitResult("fields.p", "fields", "p"),
            ],
            {"fields": {"U": "v", "p": "p"}},
        ),
        (
            [InitResult("models.algo", "models", "a")],
            {"models": {"algo": "a"}},
        ),
        (
            [InitResult("operators.mom", "operators", "m")],
            {"models": {"mom": "m"}},
        ),
        (
            [InitResult("mesh", "resource", "m")],
            {"mesh": "m"},
        ),
        (
            [InitResult("time", "resource", "r")],
            {"time": "r"},
        ),
        (
            [InitResult("custom", "resource", "c")],
            {"models": {"custom": "c"}},
        ),
        (
            [InitResult("_foam_time", "resource", "hidden")],
            {"models": {}, "time": None},
        ),
    ],
    ids=[
        "fields",
        "models",
        "operators",
        "mesh",
        "time",
        "unknown_resource",
        "init_only",
    ],
)
def test_default_router_routing(init_results, expected):
    ctx = build_context_from_results(init_results)
    assert ctx.fields == expected.get("fields", {})
    assert ctx.models == expected.get("models", {})
    assert ctx.mesh == expected.get("mesh")
    assert ctx.time == expected.get("time")


def test_write_flag_collected_into_write_fields() -> None:
    ctx = build_context_from_results(
        [
            InitResult("fields.p", "fields", "p", write=True),
            InitResult("fields.U", "fields", "v", write=True),
            InitResult("fields.UEqn", "fields", "m"),  # not flagged
        ]
    )
    assert ctx.fields == {"p": "p", "U": "v", "UEqn": "m"}
    assert ctx.write_fields == {"p", "U"}  # only write=True fields


def test_write_fields_defaults_empty() -> None:
    ctx = build_context_from_results([InitResult("fields.p", "fields", "p")])
    assert ctx.write_fields == set()


def test_build_context_routes_by_category():
    init_results = [
        InitResult("fields.U", "fields", "velocity"),
        InitResult("models.transport", "models", "transport_model"),
        InitResult("operators.laplacian", "operators", "laplacian_op"),
    ]
    ctx = build_context_from_results(init_results)
    assert ctx.fields == {"U": "velocity"}
    assert ctx.models == {
        "transport": "transport_model",
        "laplacian": "laplacian_op",
    }


def test_build_context_category_without_prefix():
    init_results = [
        InitResult("velocity", "fields", "vel_value"),
        InitResult("algo", "models", "algo_value"),
    ]
    ctx = build_context_from_results(init_results)
    assert ctx.fields == {"velocity": "vel_value"}
    assert ctx.models == {"algo": "algo_value"}


def test_unknown_category_falls_back_to_models_with_warning(caplog):
    unknown = InitResult("solver_state", "completely_unknown", "state_obj")  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING):
        ctx = build_context_from_results([unknown])

    assert ctx.models["solver_state"] == "state_obj"
    assert any(
        "solver_state" in record.message and "models" in record.message
        for record in caplog.records
    )


def test_custom_handler_can_be_registered():
    router = default_router()
    captured: list[tuple[str, str]] = []

    def turbulence_handler(builder: ContextBuilder, name: str, value: str) -> None:
        captured.append((name, value))
        builder.models[name] = value

    router.register("turbulence", turbulence_handler)

    inits = [
        InitStep(
            "k_epsilon",
            depends_on=[],
            initializer=lambda _ctx: "k_eps_obj",
            category="turbulence",  # type: ignore[arg-type]
        ),
    ]
    ctx = execute_initialization(inits, router=router)

    assert captured == [("k_epsilon", "k_eps_obj")]
    assert ctx.models == {"k_epsilon": "k_eps_obj"}


def test_execute_initialization_end_to_end():
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "mesh_obj"),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda ctx: f"U_on_{ctx['mesh']}",
            category="fields",
        ),
        InitStep(
            "models.algo",
            depends_on=[],
            initializer=lambda _ctx: "algorithm",
            category="models",
        ),
    ]
    ctx = execute_initialization(inits)
    assert isinstance(ctx, Context)
    assert ctx.mesh == "mesh_obj"
    assert ctx.fields == {"U": "U_on_mesh_obj"}
    assert ctx.models == {"algo": "algorithm"}


def test_execute_initialization_complex():
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "mesh"),
        InitStep("time", depends_on=[], initializer=lambda _ctx: "time"),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda _ctx: "velocity",
            category="fields",
        ),
        InitStep(
            "fields.p",
            depends_on=["mesh"],
            initializer=lambda _ctx: "pressure",
            category="fields",
        ),
        InitStep(
            "operators.mom",
            depends_on=["fields.U", "fields.p"],
            initializer=lambda _ctx: "momentum_op",
            category="operators",
        ),
        InitStep(
            "models.algo",
            depends_on=["operators.mom"],
            initializer=lambda _ctx: "algorithm",
            category="models",
        ),
    ]
    ctx = execute_initialization(inits)
    assert ctx.mesh == "mesh"
    assert ctx.time == "time"
    assert len(ctx.fields) == 2
    assert len(ctx.models) == 2


def test_execute_initialization_raises_structured_graph_error():
    inits = [InitStep("A", depends_on=["missing"], initializer=lambda _ctx: "a")]

    with pytest.raises(InitializationGraphError) as exc_info:
        execute_initialization(inits)

    report = exc_info.value.report
    assert not report.is_valid
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.code == "missing_dependency"
    assert diag.node_name == "A"


def test_execute_initialization_enforces_replaces_target():
    # The replacement-target check must fire on the real executed path, not only
    # inside ``_topological_sort`` (which the executed path skips by sorting with
    # ``validate_graph=False``). A ``replaces=[X]`` naming no present step raises.
    inits = [
        InitStep(
            "mesh",
            depends_on=[],
            initializer=lambda _ctx: "m",
            replaces=["nonexistent"],
        ),
    ]

    with pytest.raises(InitializationGraphError):
        execute_initialization(inits)


def test_execute_initialization_validates_only_once(monkeypatch):
    inits = [
        InitStep("mesh", depends_on=[], initializer=lambda _ctx: "mesh_obj"),
        InitStep(
            "fields.U",
            depends_on=["mesh"],
            initializer=lambda _ctx: "U",
            category="fields",
        ),
    ]

    validate_calls = {"count": 0}
    from neofoam.framework.initialization.execution import validate as original_validate

    def counting_validate(lazy_inits):
        validate_calls["count"] += 1
        return original_validate(lazy_inits)

    monkeypatch.setattr(
        "neofoam.framework.initialization.execution.validate",
        counting_validate,
    )

    ctx = execute_initialization(inits)

    assert ctx.mesh == "mesh_obj"
    assert ctx.fields == {"U": "U"}
    assert validate_calls["count"] == 1


def test_category_router_register_overrides_handler():
    router = CategoryRouter()
    seen = []
    router.register("fields", lambda b, n, v: seen.append(("first", n, v)))
    router.register("fields", lambda b, n, v: seen.append(("second", n, v)))

    builder = ContextBuilder()
    router.route(builder, InitResult("x", "fields", 1))

    assert seen == [("second", "x", 1)]
