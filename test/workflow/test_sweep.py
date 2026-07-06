# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Canvas -> sweep design -> generated workflow directory (UI-free layer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from neofoam.workflow.sweep import (
    ALL_RULE,
    DIM_NODE_TYPE,
    SETUP_RULE,
    SOLVE_RULE,
    SweepDimension,
    autowire,
    cross_product,
    dim_node,
    export_sweep,
    load_sweep,
    nodes_to_dimensions,
    rule_nodes,
    sweep_snakefile,
    validate_dimensions,
    variant_errors,
)


class _Transport(BaseModel):
    transportModel: str = "Newtonian"
    nu: float = 1e-5


class _Control(BaseModel):
    endTime: float = 1.0
    deltaT: float = 0.001


_TRANSPORT_DIM = SweepDimension(
    name="transport",
    title="transportProperties",
    schema={"type": "object", "properties": {"nu": {"type": "number"}}},
    seed={"transportModel": "Newtonian", "nu": 1e-5},
)

_CONTROL_DIM = SweepDimension(
    name="control",
    title="controlDict",
    schema={"type": "object", "properties": {"endTime": {"type": "number"}}},
    seed={"endTime": 1.0, "deltaT": 0.001},
)


def _canvas() -> list[dict[str, Any]]:
    transport = dim_node(
        "dim:transport",
        _TRANSPORT_DIM,
        entries={"nu1": {"nu": 1e-5}, "nu2": {"nu": 2e-5}},
    )
    control = dim_node("dim:control", _CONTROL_DIM)  # seeded single "base" variant
    return [transport, control, *rule_nodes(["transport", "control"])]


def test_dim_node_seeds_base_variant() -> None:
    node = dim_node("dim:transport", _TRANSPORT_DIM)
    assert node["type"] == DIM_NODE_TYPE
    assert node["data"]["dim"] == "transport"
    assert node["data"]["label"] == "transportProperties"
    assert node["data"]["entries"] == {"base": _TRANSPORT_DIM.seed}
    assert node["data"]["selected"] == "base"
    assert node["data"]["schema"] == _TRANSPORT_DIM.schema


def test_rule_nodes_default_pipeline() -> None:
    nodes = rule_nodes(["transport"])
    by_rule = {n["data"]["rule"]: n for n in nodes}
    # Registry order, `all` last on the canvas.
    assert [n["data"]["rule"] for n in nodes] == [
        "setup_mesh",
        "blockMesh",
        "snappyHexMesh",
        "checkMesh",
        SETUP_RULE,
        SOLVE_RULE,
        ALL_RULE,
    ]
    # The mesh chain is stamp-chained; setup waits on the chain's sink.
    assert by_rule["blockMesh"]["data"]["inputs"] == ["meshes/{mesh}/.staged.json"]
    assert by_rule["snappyHexMesh"]["data"]["inputs"] == [
        "meshes/{mesh}/.blockMesh.done"
    ]
    assert by_rule["setup"]["data"]["inputs"] == [
        "configs/{case}/setup.json",
        "meshes/{mesh}/.checkMesh.done",
    ]
    # Config ports: non-mesh dims on setup, the mesh dim on setup_mesh.
    assert by_rule["setup"]["data"]["cfg_dims"] == ["transport"]
    assert by_rule["setup_mesh"]["data"]["cfg_dims"] == ["mesh"]
    assert by_rule["setup"]["data"]["outputs"] == by_rule["solve"]["data"]["inputs"]
    assert by_rule["solve"]["data"]["outputs"] == by_rule["all"]["data"]["inputs"]
    assert by_rule["all"]["data"]["outputs"] == []


def test_rule_nodes_mesh_dim_is_not_a_setup_port() -> None:
    nodes = rule_nodes(["transport", "mesh"])
    by_rule = {n["data"]["rule"]: n for n in nodes}
    assert by_rule["setup"]["data"]["cfg_dims"] == ["transport"]
    assert by_rule["setup_mesh"]["data"]["cfg_dims"] == ["mesh"]


def test_rule_nodes_blockmesh_only_chain() -> None:
    nodes = rule_nodes([], enabled=["all", "setup_mesh", "blockMesh", "setup", "solve"])
    by_rule = {n["data"]["rule"]: n for n in nodes}
    assert by_rule["setup"]["data"]["inputs"][1] == "meshes/{mesh}/.blockMesh.done"


def test_autowire_connects_dims_and_rules() -> None:
    nodes = _canvas()
    edges = autowire(nodes)
    pairs = {(e["source"], e["target"]) for e in edges}
    assert ("dim:transport", "rule:setup") in pairs
    assert ("dim:control", "rule:setup") in pairs
    # The full file chain: staging -> mesh tools -> setup -> solve -> all.
    assert ("rule:setup_mesh", "rule:blockMesh") in pairs
    assert ("rule:blockMesh", "rule:snappyHexMesh") in pairs
    assert ("rule:snappyHexMesh", "rule:checkMesh") in pairs
    assert ("rule:checkMesh", "rule:setup") in pairs
    assert ("rule:setup", "rule:solve") in pairs
    assert ("rule:solve", "rule:all") in pairs
    assert len(edges) == 8  # 6 file edges + 2 cfg edges
    cfg = next(e for e in edges if e["source"] == "dim:transport")
    assert cfg["sourceHandle"] == cfg["targetHandle"] == "cfg:transport"


def test_nodes_to_dimensions_extracts_variants() -> None:
    dims = nodes_to_dimensions(_canvas())
    assert dims == {
        "transport": {"nu1": {"nu": 1e-5}, "nu2": {"nu": 2e-5}},
        "control": {"base": _CONTROL_DIM.seed},
    }


def test_nodes_to_dimensions_rejects_bad_variants() -> None:
    node = dim_node("dim:transport", _TRANSPORT_DIM, entries={"": {"nu": 1e-5}})
    with pytest.raises(ValueError, match="must not be empty"):
        nodes_to_dimensions([node])

    node = dim_node("dim:transport", _TRANSPORT_DIM)
    node["data"]["entries"] = {}
    with pytest.raises(ValueError, match="no variants defined"):
        nodes_to_dimensions([node])

    duplicated = [
        dim_node("dim:t1", _TRANSPORT_DIM, entries={"base": {}}),
        dim_node("dim:t2", _TRANSPORT_DIM, entries={"base": {}}),
    ]
    with pytest.raises(ValueError, match="duplicate variant name 'base'"):
        nodes_to_dimensions(duplicated)


def test_cross_product_names_and_order() -> None:
    rows = cross_product(
        {"transport": {"nu1": {}, "nu2": {}}, "control": {"short": {}, "long": {}}}
    )
    # Dimensions sorted (control < transport); case = variant names in that order.
    assert [r["case"] for r in rows] == [
        "long_nu1",
        "long_nu2",
        "short_nu1",
        "short_nu2",
    ]
    assert rows[0] == {"case": "long_nu1", "control": "long", "transport": "nu1"}


def test_validate_dimensions_names_offender() -> None:
    classes = {"transport": _Transport, "control": _Control}
    validate_dimensions(
        {"transport": {"ok": {"nu": "2e-5"}}}, classes
    )  # coercible: fine
    with pytest.raises(ValueError, match="'transport.bad' failed validation"):
        validate_dimensions({"transport": {"bad": {"nu": "not-a-number"}}}, classes)
    with pytest.raises(ValueError, match="unknown dimension 'other'"):
        validate_dimensions({"other": {"x": {}}}, classes)


def test_variant_errors_reports_per_variant_without_raising() -> None:
    classes = {"transport": _Transport, "control": _Control}
    # Only the failing variant appears; the message names the field.
    errors = variant_errors(
        {"transport": {"ok": {"nu": 1e-5}, "bad": {"nu": "not-a-number"}}}, classes
    )
    assert set(errors) == {"transport"}
    assert set(errors["transport"]) == {"bad"}
    assert "nu" in errors["transport"]["bad"]
    # A wholly valid sweep yields no errors.
    assert variant_errors({"transport": {"ok": {"nu": 1e-5}}}, classes) == {}
    # An unknown dimension marks its variant rather than raising.
    assert (
        "unknown dimension" in variant_errors({"nope": {"v": {}}}, classes)["nope"]["v"]
    )


def test_sweep_snakefile_header_and_includes() -> None:
    text = sweep_snakefile(
        "incompressibleFluid", "/tmp/my base", ["control", "transport", "mesh"]
    )
    assert "from neofoam.workflow.paramspace import YamlParamSpace" in text
    assert "from neofoam.workflow.rules import rules_dir" in text
    # Header globals the static .smk files consume.
    assert 'SOLVER = "incompressibleFluid"' in text
    assert 'SOLVER_CMD = "incompressiblefluid"' in text
    assert 'BASE_CASE = "/tmp/my base"' in text
    assert "SETUP_DIMS = ['control', 'transport']" in text  # mesh excluded
    assert "'blockMesh': '.staged.json'" in text
    assert 'MESH_DONE = ".checkMesh.done"' in text
    assert 'FINAL_PATTERN = "cases/{case}/done"' in text
    assert 'mesh_axis = space.keyed("mesh", out_dir="configs")' in text
    # `all` is inline (Snakemake ignores include:d rules as the implicit
    # default target); the pipeline rules come from includes.
    assert "rule all:" in text
    assert "expand(FINAL_PATTERN, case=cases)" in text
    includes = [line for line in text.splitlines() if line.startswith("include:")]
    assert includes[0].endswith('"setup_mesh.smk")')
    assert len(includes) == 6
    # The header (everything before the includes / the inline rule) is plain
    # Python — compile it to catch generator syntax bugs early.
    header_lines = []
    for line in text.splitlines():
        if line.startswith("include:") or line.startswith("rule "):
            break
        header_lines.append(line)
    compile("\n".join(header_lines), "<snakefile>", "exec")


def test_sweep_snakefile_respects_enabled_selection() -> None:
    text = sweep_snakefile(
        "incompressibleFluid",
        "/tmp/base",
        [],
        enabled=["all", "setup_mesh", "blockMesh", "setup", "solve"],
    )
    assert 'MESH_DONE = ".blockMesh.done"' in text
    includes = [line for line in text.splitlines() if line.startswith("include:")]
    assert len(includes) == 4
    assert not any("snappy" in line for line in includes)


def test_export_sweep_writes_workflow_dir(tmp_path: Path) -> None:
    dims = nodes_to_dimensions(_canvas())
    export = export_sweep(
        tmp_path / "sweep",
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions=dims,
        classes={"transport": _Transport, "control": _Control},
    )
    assert export.sweep_csv.read_text().splitlines()[0] == "case,control,transport"
    assert export.params_yaml.is_file()
    assert "include:" in export.snakefile.read_text()
    # Per-case setup configs + the implicit shared mesh variant.
    assert sorted(
        p.relative_to(export.out_dir / "configs").as_posix() for p in export.configs
    ) == [
        "base_nu1/setup.json",
        "base_nu2/setup.json",
        "mesh/base.json",
    ]
    assert (export.out_dir / "configs" / "mesh" / "base.json").read_text() == "{}\n"


def test_export_sweep_requires_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no sweep dimensions"):
        export_sweep(
            tmp_path,
            solver_name="incompressibleFluid",
            base_case=tmp_path,
            dimensions={},
            classes={},
        )


def test_load_sweep_round_trips_export(tmp_path: Path) -> None:
    classes = {"transport": _Transport, "control": _Control}
    dims = {
        "transport": {
            "nu1": {"transportModel": "Newtonian", "nu": 1e-5},
            "nu2": {"transportModel": "Newtonian", "nu": 2e-5},
        },
        "control": {"base": {"endTime": 1.0, "deltaT": 0.001}},
    }
    enabled = ["all", "setup_mesh", "blockMesh", "setup", "solve"]
    export_sweep(
        tmp_path / "sweep",
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions=dims,
        classes=classes,
        enabled=enabled,
    )

    loaded = load_sweep(tmp_path / "sweep")
    assert loaded.solver_name == "incompressibleFluid"
    assert loaded.base_case == str((tmp_path / "base").resolve())
    # Full variant payloads round-trip through params.yaml.
    assert loaded.dimensions == dims
    # Enabled rules recovered from the includes + inline `all`.
    assert set(loaded.enabled) == set(enabled)
    assert "snappyHexMesh" not in loaded.enabled  # was not enabled

    # A non-sweep directory is a clear error, not a crash.
    with pytest.raises(ValueError, match="not an exported sweep"):
        load_sweep(tmp_path)


def test_export_sweep_with_real_config_classes(tmp_path: Path) -> None:
    pytest.importorskip("pybFoam")
    from neofoam.mcp import tools
    from neofoam.mcp.registry import resolve_solver
    from neofoam.workflow.sweep_runner import config_classes_by_name

    solver = resolve_solver("incompressibleFluid")
    transport = tools.config_schema(solver, "transport_properties_config")
    classes = config_classes_by_name(solver)

    base = dict(transport.defaults)
    variant = {**base, "nu": 2e-5}
    export = export_sweep(
        tmp_path / "sweep",
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions={"transport_properties_config": {"base": base, "nu2": variant}},
        classes=classes,
    )
    assert [p.parent.name for p in export.configs] == ["base", "nu2", "mesh"]

    with pytest.raises(ValueError, match="failed validation"):
        validate_dimensions(
            {"transport_properties_config": {"bad": {**base, "nu": "not-a-number"}}},
            classes,
        )


def test_export_sweep_with_mesh_dimension(tmp_path: Path) -> None:
    pytest.importorskip("pybFoam")
    from neofoam.mcp.registry import resolve_solver
    from neofoam.workflow.sweep import validate_mesh_dimension
    from neofoam.workflow.sweep_runner import config_classes_by_name

    classes = config_classes_by_name(resolve_solver("incompressibleFluid"))
    mesh = {
        "coarse": {"block_mesh_dict_config": {"scale": 1.0}},
        "fine": {
            "block_mesh_dict_config": {"scale": 1.0},
            "snappy_hex_mesh_dict_config": {},
        },
    }
    export = export_sweep(
        tmp_path / "sweep",
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions={
            "mesh": mesh,
            "transport_properties_config": {"nu1": {"transportModel": "Newtonian"}},
        },
        classes=classes,
    )
    # The mesh dim is a sweep column (case names include the variant) but is
    # excluded from the per-case setup payloads — it applies in the mini-case.
    assert (
        export.sweep_csv.read_text().splitlines()[0]
        == "case,mesh,transport_properties_config"
    )
    rel = sorted(
        p.relative_to(export.out_dir / "configs").as_posix() for p in export.configs
    )
    assert rel == [
        "coarse_nu1/setup.json",
        "fine_nu1/setup.json",
        "mesh/coarse.json",
        "mesh/fine.json",
    ]
    import json

    setup = json.loads(
        (export.out_dir / "configs" / "coarse_nu1" / "setup.json").read_text()
    )
    assert set(setup) == {"transport_properties_config"}

    # Mesh validation names the offending variant.config.
    with pytest.raises(ValueError, match="mesh variant 'bad' names unknown config"):
        validate_mesh_dimension({"bad": {"not_a_config": {}}}, classes)
    with pytest.raises(
        ValueError, match="'bad.block_mesh_dict_config' failed validation"
    ):
        validate_mesh_dimension(
            {"bad": {"block_mesh_dict_config": {"scale": "not-a-number"}}}, classes
        )
