# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The predefined Snakemake rule library (registry, plans, packaged .smk files)."""

from __future__ import annotations

import re

import pytest

from neofoam.tooling.workflow.rules import (
    CAD_STAMP,
    DEFAULT_ENABLED,
    MESH_STAGE_STAMP,
    POST_DONE_PATTERN,
    RUN_DONE_PATTERN,
    RuleRegistry,
    RuleSpec,
    default_registry,
    rules_dir,
)


def test_default_registry_pipeline_order() -> None:
    registry = default_registry()
    assert registry.names == (
        "all",
        "setup_mesh",
        "cad_geometry",
        "blockMesh",
        "snappyHexMesh",
        "checkMesh",
        "setup",
        "solve",
        "post",
    )
    assert registry.get("blockMesh").creates_mesh
    assert registry.get("setup_mesh").consumes_mesh_dim
    assert registry.get("setup").consumes_case_dims
    # Mesh tools are the keyed chain rules, staging excluded — cad_geometry is a
    # geometry producer that runs first when enabled.
    assert [s.name for s in map(registry.get, registry.names) if s.is_mesh_tool] == [
        "cad_geometry",
        "blockMesh",
        "snappyHexMesh",
        "checkMesh",
    ]


def test_default_plan_chains_the_mesh_tools() -> None:
    plan = default_registry().plan()
    assert plan.names == DEFAULT_ENABLED
    assert plan.mesh_tool_input == {
        "blockMesh": MESH_STAGE_STAMP,
        "snappyHexMesh": ".blockMesh.done",
        "checkMesh": ".snappyHexMesh.done",
    }
    assert plan.mesh_done == ".checkMesh.done"
    assert plan.final_pattern == RUN_DONE_PATTERN


def test_plan_blockmesh_only_chain() -> None:
    plan = default_registry().plan(["all", "setup_mesh", "blockMesh", "setup", "solve"])
    assert plan.mesh_tool_input == {"blockMesh": MESH_STAGE_STAMP}
    assert plan.mesh_done == ".blockMesh.done"


def test_plan_with_post_extends_the_target() -> None:
    plan = default_registry().plan([*DEFAULT_ENABLED, "post"])
    assert plan.final_pattern == POST_DONE_PATTERN


def test_plan_validation_errors() -> None:
    registry = default_registry()
    with pytest.raises(ValueError, match=r"unknown rule\(s\) \['fluxMagic'\]"):
        registry.plan([*DEFAULT_ENABLED, "fluxMagic"])
    with pytest.raises(ValueError, match="'solve' is required"):
        registry.plan(["all", "setup_mesh", "blockMesh", "setup"])
    # setup_mesh stages a CLEAN variant dir — the chain must start by creating
    # a mesh, not by consuming one.
    with pytest.raises(ValueError, match="starts at 'snappyHexMesh'"):
        registry.plan(["all", "setup_mesh", "snappyHexMesh", "setup", "solve"])
    with pytest.raises(ValueError, match="at least one mesh tool"):
        registry.plan(["all", "setup_mesh", "setup", "solve"])


def test_rule_plan_get_rejects_rule_outside_plan() -> None:
    plan = default_registry().plan(["all", "setup_mesh", "blockMesh", "setup", "solve"])
    with pytest.raises(KeyError, match="not part of this plan"):
        plan.get("post")


def test_registry_rejects_duplicates_and_unknown() -> None:
    spec = default_registry().get("solve")
    with pytest.raises(ValueError, match="duplicate rule names"):
        RuleRegistry([spec, spec])
    with pytest.raises(KeyError, match="unknown rule 'x'"):
        default_registry().get("x")


def test_packaged_smk_files_exist_and_define_their_rule() -> None:
    registry = default_registry()
    directory = rules_dir()
    for name in registry.names:
        spec: RuleSpec = registry.get(name)
        if not spec.smk_file:  # `all` is emitted inline by the generator
            continue
        path = directory / spec.smk_file
        assert path.is_file(), f"missing packaged rule file {spec.smk_file}"
        text = path.read_text()
        assert re.search(rf"^rule {spec.name}:$", text, re.M), (
            f"{spec.smk_file} does not define 'rule {spec.name}:'"
        )
        # Every declared output pattern appears in the rule file. Mesh-chain
        # rules build their output from the header's MESH_STEM (a composite
        # cad × mesh dir at runtime), so assert MESH_STEM + the stamp suffix.
        for out in spec.outputs:
            if out.startswith("meshes/{mesh}/"):
                suffix = out[len("meshes/{mesh}") :]
                assert "MESH_STEM" in text, f"{spec.smk_file} lacks MESH_STEM"
                assert suffix in text, f"{spec.smk_file} does not produce {suffix}"
            else:
                assert out in text, f"{spec.smk_file} does not produce {out}"


def test_cad_geometry_registered_and_prepends_the_mesh_chain() -> None:
    registry = default_registry()
    assert "cad_geometry" in registry.names
    spec = registry.get("cad_geometry")
    assert spec.is_mesh_tool
    assert spec.produces_geometry
    assert not spec.creates_mesh
    # Opt-in, like post: not on by default so mesh-only sweeps are unchanged.
    assert "cad_geometry" not in DEFAULT_ENABLED

    # Enabled, cad_geometry runs first in the mesh chain; blockMesh waits on its
    # stamp (which lives in the STL directory it populates).
    plan = registry.plan([*DEFAULT_ENABLED, "cad_geometry"])
    assert plan.mesh_tool_input["cad_geometry"] == MESH_STAGE_STAMP
    assert plan.mesh_tool_input["blockMesh"] == CAD_STAMP
    assert spec.stamp == CAD_STAMP
    assert CAD_STAMP.endswith(".cad.done")
    # The rest of the chain is unchanged; setup still waits on checkMesh.
    assert plan.mesh_done == ".checkMesh.done"


def test_cad_geometry_smk_matches_spec() -> None:
    text = (rules_dir() / "cad_geometry.smk").read_text()
    assert re.search(r"^rule cad_geometry:$", text, re.M)
    # The output composes the header's MESH_STEM (cad × mesh) with the cad stamp.
    assert "MESH_STEM" in text
    assert "/constant/triSurface/.cad.done" in text
    assert "sweep_runner cad" in text
    # References the header globals it consumes.
    for symbol in ("CAD_MODEL", "_mesh_case"):
        assert symbol in text


def test_smk_files_document_consumed_globals() -> None:
    # The .smk files reference plain header globals; each names them in its
    # comment banner so generated-Snakefile drift is visible at review time.
    directory = rules_dir()
    expected = {
        "setup_mesh.smk": ("SOLVER", "BASE_CASE", "MESH_STEM"),
        "block_mesh.smk": ("MESH_TOOL_INPUT", "MESH_STEM"),
        "setup.smk": ("_mesh_dir_of", "MESH_DONE"),
        "solve.smk": ("SOLVER_CMD",),
    }
    for filename, globals_used in expected.items():
        text = (directory / filename).read_text()
        for symbol in globals_used:
            assert symbol in text, f"{filename} should reference {symbol}"
