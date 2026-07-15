# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Canvas -> sweep design -> generated workflow directory (UI-free layer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from neofoam.tooling.workflow.rules import default_registry
from neofoam.tooling.workflow.sweep import (
    ALL_RULE,
    DIM_NODE_TYPE,
    SETUP_RULE,
    SOLVE_RULE,
    Sweep,
    SweepDimension,
    autowire,
    cross_product,
    dim_node,
    export_sweep,
    load_sweep,
    nodes_to_dimensions,
    rule_nodes,
    sweep_snakefile,
    validate_cad_dimension,
    validate_dimensions,
    validate_mesh_dimension,
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


def test_variant_errors_reports_mesh_dimension_failures() -> None:
    # The mesh dimension is config-name-keyed: each variant maps config names to
    # payloads validated against their own class. Only failing variants appear.
    classes = {"transport": _Transport, "control": _Control}
    dims = {
        "mesh": {
            "good": {"transport": {"nu": 1e-5}},
            "not_mapping": [],
            "unknown_cfg": {"nope": {}},
            "invalid": {"transport": {"nu": "not-a-number"}},
        }
    }
    errors = variant_errors(dims, classes)
    assert set(errors) == {"mesh"}
    assert set(errors["mesh"]) == {"not_mapping", "unknown_cfg", "invalid"}
    assert errors["mesh"]["not_mapping"] == "must map config names to payloads"
    assert "unknown config 'nope'" in errors["mesh"]["unknown_cfg"]
    assert errors["mesh"]["invalid"].startswith("transport:")
    assert "nu" in errors["mesh"]["invalid"]


def test_validate_mesh_dimension_rejects_non_mapping_payload() -> None:
    classes = {"transport": _Transport}
    with pytest.raises(ValueError, match="must map config names"):
        validate_mesh_dimension({"bad": ["a", "b"]}, classes)


def test_cross_product_rejects_colliding_case_names() -> None:
    # Distinct variant tuples collapse to the same case name when variant names
    # contain '_': with dims a<b, (p, q_r) and (p_q, r) both give 'p_q_r'.
    with pytest.raises(ValueError, match="collision"):
        cross_product({"a": {"p": {}, "p_q": {}}, "b": {"q_r": {}, "r": {}}})


def test_export_sweep_aborts_without_partial_write_on_collision(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="collision"):
        export_sweep(
            out,
            solver_name="incompressibleFluid",
            base_case=tmp_path / "base",
            dimensions={
                "control": {"p": {}, "p_q": {}},
                "transport": {"q_r": {}, "r": {}},
            },
            classes={"transport": _Transport, "control": _Control},
        )
    # No workflow files were written — the collision aborted before any write.
    assert not (out / "sweep.csv").exists()
    assert not (out / "params.yaml").exists()
    assert not (out / "Snakefile").exists()


def test_load_sweep_reads_the_sidecar_not_the_snakefile(tmp_path: Path) -> None:
    classes = {"transport": _Transport, "control": _Control}
    dims = {
        "transport": {"nu1": {"transportModel": "Newtonian", "nu": 1e-5}},
        "control": {"base": {"endTime": 1.0, "deltaT": 0.001}},
    }
    sweep_dir = tmp_path / "sweep"
    export_sweep(
        sweep_dir,
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions=dims,
        classes=classes,
    )

    # sweep.meta.json is the read-back contract, so corrupting the generated
    # Snakefile header does not change what load_sweep recovers.
    snakefile = sweep_dir / "Snakefile"
    snakefile.write_text(snakefile.read_text().replace("SOLVER = ", "SOLVERX = "))
    loaded = load_sweep(sweep_dir)
    assert loaded.solver_name == "incompressibleFluid"
    assert loaded.base_case == str((tmp_path / "base").resolve())

    # A missing sidecar still loads via params.yaml, with header defaults.
    (sweep_dir / "sweep.meta.json").unlink()
    fallback = load_sweep(sweep_dir)
    assert fallback.dimensions == dims
    assert fallback.base_case == ""
    assert fallback.solver_name == "incompressibleFluid"
    assert fallback.enabled == ["all"]


def test_sweep_snakefile_header_and_includes() -> None:
    text = sweep_snakefile(
        "incompressibleFluid", "/tmp/my base", ["control", "transport", "mesh"]
    )
    assert "from neofoam.tooling.workflow.paramspace import YamlParamSpace" in text
    assert "from neofoam.tooling.workflow.rules import rules_dir" in text
    # Header globals the static .smk files consume.
    assert 'SOLVER = "incompressibleFluid"' in text
    assert 'SOLVER_CMD = "incompressiblefluid"' in text
    assert 'BASE_CASE = "/tmp/my base"' in text
    assert "SETUP_DIMS = ['control', 'transport']" in text  # mesh excluded
    assert "'blockMesh': '.staged.json'" in text
    assert 'MESH_DONE = ".checkMesh.done"' in text
    assert 'FINAL_PATTERN = "cases/{case}/done"' in text
    assert 'mesh_axis = space.keyed("mesh", out_dir="configs")' in text
    # Mesh-only sweep: MESH_STEM stays the single-{mesh} dir.
    assert 'MESH_STEM = "meshes/{mesh}"' in text
    assert "def _mesh_dir_of(wc):" in text
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
    # Enabled rules recovered from the sweep.meta.json sidecar.
    assert set(loaded.enabled) == set(enabled)
    assert "snappyHexMesh" not in loaded.enabled  # was not enabled

    # A non-sweep directory is a clear error, not a crash.
    with pytest.raises(ValueError, match="not an exported sweep"):
        load_sweep(tmp_path)


def test_export_sweep_with_real_config_classes(tmp_path: Path) -> None:
    from neofoam.mcp import tools
    from neofoam.mcp.registry import resolve_solver
    from neofoam.tooling.workflow.sweep_runner import config_classes_by_name

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


_CAD = {
    "cad": {
        "model": "geometry/design.FCStd",
        "variants": {"d6": {"tube_d": 6.0}, "d8": {"tube_d": 8.0}},
    }
}


def test_validate_cad_dimension_accepts_numbers_rejects_rest() -> None:
    validate_cad_dimension({"d6": {"tube_d": 6.0, "n_tubes": 8}})  # ints + floats: fine
    with pytest.raises(
        ValueError, match=r"'cad\.d6' failed validation.*must map parameter aliases"
    ):
        validate_cad_dimension({"d6": ["not", "a", "map"]})
    with pytest.raises(
        ValueError, match=r"'cad\.d6' failed validation: tube_d: must be a number"
    ):
        validate_cad_dimension({"d6": {"tube_d": "wide"}})
    # A bool is not accepted as a numeric parameter.
    with pytest.raises(ValueError, match="must be a number"):
        validate_cad_dimension({"d6": {"flag": True}})


def test_export_sweep_with_cad_axis(tmp_path: Path) -> None:
    export = export_sweep(
        tmp_path / "sweep",
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions={"transport": {"nu1": {"nu": 1e-5}, "nu2": {"nu": 2e-5}}},
        classes={"transport": _Transport},
        cad=_CAD,
    )
    # A per-variant cad config is materialized next to configs/mesh.
    assert (export.out_dir / "configs" / "cad" / "d6.json").is_file()
    assert (export.out_dir / "configs" / "cad" / "d8.json").is_file()

    snake = export.snakefile.read_text()
    assert 'CAD_MODEL = "geometry/design.FCStd"' in snake
    assert 'cad_axis = space.keyed("cad", out_dir="configs")' in snake
    includes = [line for line in snake.splitlines() if line.startswith("include:")]
    # The cad include comes before blockMesh in the mesh chain.
    assert any("cad_geometry.smk" in line for line in includes)
    cad_idx = next(i for i, line in enumerate(includes) if "cad_geometry.smk" in line)
    block_idx = next(i for i, line in enumerate(includes) if "block_mesh.smk" in line)
    assert cad_idx < block_idx

    # Case count = cad(2) × config(2) = 4.
    rows = export.sweep_csv.read_text().splitlines()
    assert len(rows) - 1 == 4

    # The sweep round-trips: params.yaml keeps the cad variants, the header the
    # model path.
    loaded = load_sweep(export.out_dir)
    assert loaded.cad_model == "geometry/design.FCStd"
    assert set(loaded.dimensions["cad"]) == {"d6", "d8"}
    # cad is excluded from the per-case setup payloads (it applies upstream).
    import json

    setup = json.loads(
        next((export.out_dir / "configs").glob("*/setup.json")).read_text()
    )
    assert set(setup) == {"transport"}


def test_sweep_snakefile_cad_composes_mesh_stem() -> None:
    # With a CAD axis the mesh mini-case dir composes cad × mesh; the header's
    # MESH_STEM (used verbatim as the mesh rules' output:) and the _mesh_dir_of
    # helper carry the composite.
    text = sweep_snakefile(
        "incompressibleFluid",
        "/tmp/base",
        ["cad", "mesh", "transport"],
        cad_model="design.FCStd",
    )
    assert 'MESH_STEM = "meshes/{cad}__{mesh}"' in text
    assert 'return f"meshes/{cad_axis.of(wc)}__{mesh_axis.of(wc)}"' in text
    assert 'return f"meshes/{wc.cad}__{wc.mesh}"' in text


def test_export_sweep_rejects_composite_mesh_key_collision(tmp_path: Path) -> None:
    # A variant name may contain '__', so distinct (cad, mesh) pairs can collapse
    # onto one meshes/{cad}__{mesh} dir: cad 'a' + mesh 'b__c' and cad 'a__b' +
    # mesh 'c' both give 'a__b__c'. This must abort before any file is written.
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="mesh-key collision"):
        export_sweep(
            out,
            solver_name="incompressibleFluid",
            base_case=tmp_path / "base",
            dimensions={
                "mesh": {"c": {}, "b__c": {}},
                "transport": {"nu1": {"nu": 1e-5}},
            },
            classes={"transport": _Transport},
            cad={
                "cad": {
                    "model": "m.FCStd",
                    "variants": {"a": {"r": 1.0}, "a__b": {"r": 2.0}},
                }
            },
        )
    assert not (out / "sweep.csv").exists()
    assert not (out / "Snakefile").exists()


def test_export_sweep_cad_times_mesh_case_count(tmp_path: Path) -> None:
    # The CAD axis composes with the mesh axis: cad × mesh × config.
    export = export_sweep(
        tmp_path / "sweep",
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions={
            "mesh": {"coarse": {}, "fine": {}},
            "transport": {"nu1": {"nu": 1e-5}},
        },
        classes={"transport": _Transport},
        cad=_CAD,
    )
    rows = export.sweep_csv.read_text().splitlines()
    # cad(2) × mesh(2) × transport(1) = 4.
    assert len(rows) - 1 == 4
    assert export.sweep_csv.read_text().splitlines()[0] == "case,cad,mesh,transport"


def test_export_sweep_with_mesh_dimension(tmp_path: Path) -> None:
    from neofoam.mcp.registry import resolve_solver
    from neofoam.tooling.workflow.sweep import validate_mesh_dimension
    from neofoam.tooling.workflow.sweep_runner import config_classes_by_name

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
    with pytest.raises(
        ValueError,
        match=r"'mesh\.bad' failed validation: unknown config 'not_a_config'",
    ):
        validate_mesh_dimension({"bad": {"not_a_config": {}}}, classes)
    with pytest.raises(
        ValueError, match=r"'mesh\.bad' failed validation: block_mesh_dict_config"
    ):
        validate_mesh_dimension(
            {"bad": {"block_mesh_dict_config": {"scale": "not-a-number"}}}, classes
        )


# -- the Sweep façade -------------------------------------------------------
# One object owns the round-trip: its derived views (.rows / .plan / .snakefile)
# agree with the free functions it composes, and .export / .load round-trip.

_SWEEP_DIMS = {
    "transport": {
        "nu1": {"transportModel": "Newtonian", "nu": 1e-5},
        "nu2": {"transportModel": "Newtonian", "nu": 2e-5},
    },
    "control": {"base": {"endTime": 1.0, "deltaT": 0.001}},
}
_SWEEP_CLASSES = {"transport": _Transport, "control": _Control}


def _sweep(**kw: Any) -> Sweep:
    args: dict[str, Any] = dict(
        dimensions=_SWEEP_DIMS,
        solver_name="incompressibleFluid",
        base_case="/tmp/base",
        classes=_SWEEP_CLASSES,
    )
    args.update(kw)
    return Sweep(**args)


def test_sweep_derived_views_match_free_functions() -> None:
    sweep = _sweep()
    assert sweep.rows == cross_product(_SWEEP_DIMS)
    assert sweep.plan.names == default_registry().plan().names
    assert sweep.snakefile == sweep_snakefile(
        "incompressibleFluid", "/tmp/base", sorted(_SWEEP_DIMS)
    )


def test_sweep_export_matches_export_sweep(tmp_path: Path) -> None:
    from_object = _sweep(base_case=tmp_path / "base").export(tmp_path / "obj")
    from_func = export_sweep(
        tmp_path / "fn",
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions=_SWEEP_DIMS,
        classes=_SWEEP_CLASSES,
    )
    assert from_object.sweep_csv.read_text() == from_func.sweep_csv.read_text()
    assert from_object.snakefile.read_text() == from_func.snakefile.read_text()
    assert from_object.params_yaml.read_text() == from_func.params_yaml.read_text()


def test_sweep_load_round_trips(tmp_path: Path) -> None:
    enabled = ["all", "setup_mesh", "blockMesh", "setup", "solve"]
    _sweep(base_case=tmp_path / "base", enabled=enabled).export(tmp_path / "sweep")

    loaded = Sweep.load(tmp_path / "sweep")
    assert loaded.solver_name == "incompressibleFluid"
    assert loaded.base_case == str((tmp_path / "base").resolve())
    assert loaded.dimensions == _SWEEP_DIMS
    assert set(loaded.enabled or []) == set(enabled)
    # Derived views work on a loaded sweep (no config classes needed).
    assert loaded.rows == cross_product(_SWEEP_DIMS)
    assert "snappyHexMesh" not in loaded.plan.names


def test_sweep_validates_on_export(tmp_path: Path) -> None:
    bad = Sweep(
        dimensions={"transport": {"bad": {"nu": "not-a-number"}}},
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        classes=_SWEEP_CLASSES,
    )
    with pytest.raises(ValueError, match="failed validation"):
        bad.export(tmp_path / "out")
