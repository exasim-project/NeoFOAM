# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from neofoam.agent.case_fill import load_case_from_disk
from neofoam.framework.solver.configurations import configurations
from neofoam.framework.tools.graph import PreprocessConfig
from neofoam.framework.validation import checks as checks_mod
from neofoam.io import write_configs
from neofoam.mcp import tools
from neofoam.mcp.dto import (
    CaseSpecDTO,
    CaseTextDTO,
    ConfigInfoDTO,
    ConfigSchemaDTO,
    ModelEntryDTO,
)
from neofoam.mcp.registry import resolve_solver
from neofoam.mcp.tools import ALL_TOOL_NAMES
from neofoam.postprocess import PostProcessConfig
from neofoam.preprocess import SetFieldsConfig
from neofoam.solver.incompressibleFluid.incompressibleFluid import (
    incompressibleFluid,
)
from neofoam.solver.incompressibleFluid.models.boussinesq import GravityConfig
from neofoam.tooling import CaseAccessError, Workspace
from neofoam.tooling.workflow.patch_set import PatchSet

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CASE = REPO_ROOT / "test" / "solver" / "incompressibleFluid" / "val_pitzDaily"
CLEAN_CASE = REPO_ROOT / "test" / "setup_pimple"
TUBE_BANK = Path(__file__).parent / "cases" / "tube_bank"


@pytest.fixture
def solver() -> Any:
    return resolve_solver("incompressibleFluid")


def test_all_tool_names_match_the_spec_literal_list() -> None:
    assert ALL_TOOL_NAMES == (
        "list_solvers",
        "model_catalog",
        "tool_catalog",
        "list_configs",
        "config_schema",
        "case_spec_schema",
        "manifest_schema",
        "workspace_info",
        "import_geometry",
        "case_patches",
        "build_mesh_inputs",
        "save_preprocess",
        "validate_case",
        "read_case",
        "load_case",
        "save_case",
        "post_catalog",
        "save_post",
    )


def test_list_solvers_lists_incompressible_fluid() -> None:
    assert "incompressibleFluid" in tools.list_solvers()


def test_tool_catalog_lists_tools_with_step_schema_and_config(solver: Any) -> None:
    """Each preprocessing tool is catalogued with its entry schema + the config it reads."""
    catalog = tools.tool_catalog(solver)
    by_name = {t.name: t for t in catalog}
    assert {"blockMesh", "snappyHexMesh", "checkMesh"} <= set(by_name)
    # the preprocess.yaml entry schema (its ``tool`` key + options)
    block = by_name["blockMesh"]
    assert "properties" in block.step_schema and "tool" in block.step_schema["properties"]
    # blockMesh reads system/blockMeshDict, written by BlockMeshDictConfig
    assert block.dict_file == "system/blockMeshDict"
    assert block.config == "BlockMeshDictConfig"
    assert by_name["snappyHexMesh"].config == "SnappyHexMeshDictConfig"
    # checkMesh reads no dict
    assert by_name["checkMesh"].dict_file is None and by_name["checkMesh"].config is None


def test_model_catalog_is_case_free_with_expected_names(solver: Any) -> None:
    entries = tools.model_catalog(solver)
    assert all(isinstance(e, ModelEntryDTO) for e in entries)
    names = {e.name for e in entries}
    assert {"courant", "maxDeltaT", "boussinesq"} <= names
    required = {e.name for e in entries if e.required}
    assert {"Pimple", "Newtonian", "laminar"} <= required
    # Serialized owned configs are class-name strings, no class repr.
    text = ModelEntryDTO(
        name="x",
        label="x",
        required=True,
        dicts=entries[0].dicts,
        fields=entries[0].fields,
    ).model_dump_json()
    assert "ModelMetaclass" not in text


def test_list_configs_includes_control_dict_with_file(solver: Any) -> None:
    configs = tools.list_configs(solver)
    assert all(isinstance(c, ConfigInfoDTO) for c in configs)
    cd = next(c for c in configs if c.cls_name == "ControlDictConfig")
    assert cd.file == "system/controlDict"


def test_config_schema_has_schema_ui_and_defaults(solver: Any) -> None:
    dto = tools.config_schema(solver, "ControlDictConfig")
    assert isinstance(dto, ConfigSchemaDTO)
    assert dto.json_schema and "properties" in dto.json_schema
    assert dto.defaults["application"] == "pimpleFoam"
    assert isinstance(dto.ui_schema, dict)


def test_config_schema_unknown_name_errors(solver: Any) -> None:
    with pytest.raises(ValueError):
        tools.config_schema(solver, "NoSuchConfig")


def test_config_schema_accepts_snake_case_name(solver: Any) -> None:
    """The snake_case name list_configs advertises round-trips into config_schema."""
    dto = tools.config_schema(solver, "control_dict_config")
    assert dto.json_schema and "properties" in dto.json_schema


def test_case_spec_schema_describes_the_save_case_envelope(solver: Any) -> None:
    """case_spec_schema lists the snake-case config keys save_case accepts."""
    dto = tools.case_spec_schema(solver)
    assert dto.json_schema and "properties" in dto.json_schema
    # the keys are exactly the schema's properties, and cover the always-needed configs
    assert set(dto.config_keys) == set(dto.json_schema["properties"])
    assert {
        "control_dict_config",
        "transport_properties_config",
        "pimple_fv_schemes",
    } <= set(dto.config_keys)
    # every field is optional (fill only the case's subset) → no required keys
    assert dto.json_schema.get("required", []) == []


def test_manifest_schema_lists_geometry_source_as_required() -> None:
    """manifest_schema is self-describing: the manifest's required keys are surfaced."""
    dto = tools.manifest_schema()
    assert dto.json_schema and "properties" in dto.json_schema
    assert "geometry_source" in dto.required
    assert {"bbox", "location_in_mesh", "length_scale"} <= set(dto.required)
    assert "geometry_source" in dto.json_schema["properties"]


def test_case_patches_reads_staged_manifest(tmp_path: Path) -> None:
    """case_patches returns the boundary patches (name + role) from the manifest."""
    manifest = TUBE_BANK / "manifest.json"
    PatchSet.load(manifest).save(tmp_path / "manifest.json")
    patches = tools.case_patches(str(tmp_path))
    assert {p.name: p.role for p in patches} == {
        "inlet": "inlet",
        "outlet": "outlet",
        "walls": "wall",
        "tubes": "wall",
        "frontBack": "empty",
    }


def test_case_patches_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tools.case_patches(str(tmp_path))  # dir exists but has no manifest.json


def test_build_mesh_inputs_writes_mesh_dicts_from_manifest(tmp_path: Path) -> None:
    """build_mesh_inputs renders the mesh dicts + preprocess enable-list from the
    staged manifest (no hand-authored blockMeshDict/preprocess.yaml)."""
    ps = PatchSet.load(TUBE_BANK / "manifest.json")
    ps.save(tmp_path / "manifest.json")

    dto = tools.build_mesh_inputs(str(tmp_path))

    assert "system/blockMeshDict" in dto.written
    assert "system/preprocess.yaml" in dto.written
    assert (tmp_path / "system" / "blockMeshDict").is_file()
    # preprocess.yaml is a YAMLStrategy config — it must actually land on disk
    # (the write_configs YAML merged-write path).
    assert (tmp_path / "system" / "preprocess.yaml").is_file()
    # This manifest has a snappy surface (``tubes``), so a snappyHexMeshDict is written.
    has_snappy = any(p.is_snappy_surface for p in ps.patches)
    assert dto.has_snappy == has_snappy
    assert ("system/snappyHexMeshDict" in dto.written) == has_snappy


def test_build_mesh_inputs_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tools.build_mesh_inputs(str(tmp_path))  # dir exists but has no manifest.json


def _stage_mesh_dicts_with_u(tmp_path: Path, frontback_bc: str) -> Any:
    """Stage the tube-bank mesh dicts + a 0/U whose frontBack BC is ``frontback_bc``.

    The blockMeshDict / snappyHexMeshDict are checked-in fixtures (a constraint
    ``frontBack`` patch is what validate_case cross-checks against 0/U); only the U
    field varies per test, so the validator sees a real staged case.
    """
    ps = PatchSet.load(TUBE_BANK / "manifest.json")
    system = tmp_path / "system"
    system.mkdir(parents=True, exist_ok=True)
    for name in ("blockMeshDict", "snappyHexMeshDict"):
        shutil.copy(TUBE_BANK / "system" / name, system / name)
    u = configurations(incompressibleFluid)["UFieldConfig"](
        boundaryField={
            p.name: {"type": frontback_bc if p.role.value == "empty" else "zeroGradient"}
            for p in ps.patches
        }
    )
    write_configs([u], case_dir=tmp_path)


def test_validate_case_flags_constraint_patch_mismatch(solver: Any, tmp_path: Path) -> None:
    """A BC type that doesn't match a constraint mesh patch is an error with a fix."""
    _stage_mesh_dicts_with_u(tmp_path, frontback_bc="empty")  # mesh patch is symmetry
    report = tools.validate_case(solver, str(tmp_path))
    assert not report.ok
    hit = next(
        f
        for f in report.findings
        if f.file == "0/U" and "frontBack" in f.message and "symmetry" in f.message
    )
    assert hit.fix and "symmetry" in hit.fix


def test_validate_case_accepts_matching_constraint_patch(solver: Any, tmp_path: Path) -> None:
    """With the correct symmetry BC, there is no patch-mismatch finding for frontBack."""
    _stage_mesh_dicts_with_u(tmp_path, frontback_bc="symmetry")
    report = tools.validate_case(solver, str(tmp_path))
    assert not any("frontBack" in f.message for f in report.findings)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("FoamFile{ version 2.0; format ascii; class dictionary; object x; }\n" + body)


def test_validate_case_requires_constant_g_when_boussinesq(solver: Any, tmp_path: Path) -> None:
    """Boussinesq (beta+TRef in transportProperties) without constant/g is an error."""
    _write(tmp_path / "constant" / "transportProperties", "beta 3e-3;\nTRef 300;\n")
    report = tools.validate_case(solver, str(tmp_path))
    hit = next(f for f in report.findings if f.file == "constant/g")
    assert hit.level == "error" and hit.fix and "value (0 -9.81 0)" in hit.fix
    # authoring the config clears the finding
    write_configs([GravityConfig()], case_dir=tmp_path)
    assert not any(
        f.file == "constant/g" for f in tools.validate_case(solver, str(tmp_path)).findings
    )


def test_validate_case_buoyant_p_needs_no_pfinal(solver: Any, tmp_path: Path) -> None:
    """In a Boussinesq case the vestigial 'p' solver needs no 'pFinal' (p_rgh is solved).

    The buoyantBoussinesq tutorials ship U/UFinal + p_rgh/p_rghFinal + a plain 'p'
    solver with NO pFinal — the validator must not demand one; a genuinely missing
    Final (here 'U' without 'UFinal') is still flagged.
    """
    _write(tmp_path / "constant" / "transportProperties", "beta 3e-3;\nTRef 300;\n")
    _write(
        tmp_path / "system" / "fvSolution",
        "solvers{ p{ solver GAMG; smoother GaussSeidel; }\n"
        "  p_rgh{ solver GAMG; smoother GaussSeidel; }\n"
        "  p_rghFinal{ solver GAMG; smoother GaussSeidel; }\n"
        "  U{ solver smoothSolver; } }\n",
    )
    findings = tools.validate_case(solver, str(tmp_path)).findings
    msgs = [f.message for f in findings]
    assert not any("pFinal" in m for m in msgs)  # p is vestigial → no pFinal needed
    assert any("UFinal" in m for m in msgs)  # a real missing Final is still flagged


def test_validate_case_flags_wall_function_in_laminar_case(solver: Any, tmp_path: Path) -> None:
    """A ``*WallFunction`` BC is invalid when turbulence is laminar; fix → calculated."""
    _write(tmp_path / "constant" / "turbulenceProperties", "simulationType laminar;\n")
    _write(
        tmp_path / "0" / "alphat",
        "boundaryField{ walls{ type compressible::alphatWallFunction; value uniform 0; }\n"
        "inlet{ type calculated; value uniform 0; } }\n",
    )
    report = tools.validate_case(solver, str(tmp_path))
    hit = next(f for f in report.findings if f.file == "0/alphat")
    assert "walls" in hit.message and "laminar" in hit.message
    assert hit.fix and "calculated" in hit.fix
    # 'calculated' on inlet is fine — only the wall-function patch is flagged
    assert sum(1 for f in report.findings if f.file == "0/alphat") == 1


def test_validate_case_errors_when_a_boundary_type_cannot_be_read(
    solver: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable boundary type is an error, never a silently skipped check.

    Reproduces the false-success shape without depending on the backend raising:
    the reader reports the leaf as Unreadable, and validate_case must surface it.
    """

    def fake_read_section(path: Path, section: str) -> Any:
        if section == "boundaryField" and path.name == "U":
            return {"frontBack": {"type": checks_mod.Unreadable(reason="not a single string")}}
        return {}

    monkeypatch.setattr(checks_mod, "read_section", fake_read_section)
    report = tools.validate_case(solver, str(tmp_path))

    assert not report.ok
    hit = next(f for f in report.findings if f.file == "0/U" and "frontBack" in f.message)
    assert hit.level == "error" and "could not be read" in hit.message


def test_validate_case_reads_boundary_through_dictread_without_truncation(
    solver: Any, tmp_path: Path
) -> None:
    """A field carrying a vector `value` leaf still gets its constraint mismatch flagged.

    Guards the extraction: a non-scalar sibling leaf must not truncate the
    boundaryField and hide the empty-vs-symmetry error.
    """
    _stage_mesh_dicts_with_u(tmp_path, frontback_bc="empty")  # mesh patch is symmetry
    _write(
        tmp_path / "0" / "U",
        "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
        "boundaryField{ frontBack{ type empty; }\n"
        "  inlet{ type fixedValue; value uniform (1 0 0); } }\n",
    )
    report = tools.validate_case(solver, str(tmp_path))
    assert any(
        f.file == "0/U" and "frontBack" in f.message and "symmetry" in f.message
        for f in report.findings
    )


def test_read_case_returns_file_texts(solver: Any) -> None:
    dto = tools.read_case(solver, str(SOURCE_CASE))
    assert isinstance(dto, CaseTextDTO)
    assert "system/controlDict" in dto.files
    assert dto.files["system/controlDict"].strip()


def test_load_case_returns_case_spec_dump(solver: Any) -> None:
    dto = tools.load_case(solver, str(SOURCE_CASE))
    assert isinstance(dto, CaseSpecDTO)
    assert dto.values["control_dict_config"] is not None


def test_read_case_missing_dir_raises_naming_path(solver: Any, tmp_path: Path) -> None:
    missing = tmp_path / "no_such_case"
    with pytest.raises(ValueError, match=str(missing)):
        tools.read_case(solver, str(missing))


def test_load_case_missing_dir_raises_naming_path(solver: Any, tmp_path: Path) -> None:
    missing = tmp_path / "no_such_case"
    with pytest.raises(ValueError, match=str(missing)):
        tools.load_case(solver, str(missing))


def test_save_case_validates_then_writes(solver: Any, tmp_path: Path) -> None:
    spec_dump = load_case_from_disk(SOURCE_CASE, solver=solver).model_dump()

    result = tools.save_case(solver, spec_dump, str(tmp_path))

    assert result.written, "no files reported as written"
    for rel in ("system/controlDict", "constant/transportProperties"):
        assert (tmp_path / rel).exists()
    assert result.case_spec["control_dict_config"] is not None


def test_save_case_rejects_malformed_without_writing(solver: Any, tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    sentinel = target / "keepme.txt"
    sentinel.write_text("pre-existing")

    with pytest.raises(ValueError):
        tools.save_case(solver, {"control_dict_config": "not a dict"}, str(target))

    # No partial write: the sentinel is untouched and nothing new was created.
    assert sentinel.read_text() == "pre-existing"
    assert list(target.iterdir()) == [sentinel]


def test_save_case_rejects_an_escaping_target(solver: Any, tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    with pytest.raises(CaseAccessError):
        tools.save_case(solver, {}, "../escape", workspace=ws)


def test_workspace_info_reports_unconfined_by_default() -> None:
    info = tools.workspace_info()
    assert info.confined is False and info.root is None


def test_workspace_info_reports_the_active_root_when_confined(tmp_path: Path) -> None:
    info = tools.workspace_info(workspace=Workspace.at(tmp_path))
    assert info.confined is True
    assert info.root == str(tmp_path.resolve())


def _manifest_dict() -> dict[str, Any]:
    """A minimal two-patch PatchSet dump (no case_dir — import_geometry sets it)."""
    return {
        "geometry_source": "unit_test",
        "bbox": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        "location_in_mesh": [0.5, 0.5, 0.5],
        "length_scale": 0.1,
        "patches": [
            {"name": "inlet", "stl": "constant/triSurface/inlet.stl", "role": "inlet"},
            {
                "name": "frontBack",
                "stl": "constant/triSurface/frontBack.stl",
                "role": "empty",
            },
        ],
    }


def test_import_geometry_writes_manifest_read_back_by_case_patches(
    tmp_path: Path,
) -> None:
    """import_geometry stages the manifest; case_patches then reads its patches (F1)."""
    tri = tmp_path / "constant" / "triSurface"
    tri.mkdir(parents=True)
    for name in ("inlet", "frontBack"):
        (tri / f"{name}.stl").write_text("solid\nendsolid\n")

    result = tools.import_geometry(str(tmp_path), _manifest_dict())

    assert Path(result.manifest) == tmp_path / "manifest.json"
    assert (tmp_path / "manifest.json").is_file()
    assert {p.name: p.role for p in result.patches} == {
        "inlet": "inlet",
        "frontBack": "empty",
    }
    # the hand-off round-trips: case_patches reads what import_geometry wrote
    assert {p.name for p in tools.case_patches(str(tmp_path))} == {"inlet", "frontBack"}


def test_import_geometry_stages_stls_from_a_source_dir(tmp_path: Path) -> None:
    """With stl_source_dir the STLs are copied into constant/triSurface (F1)."""
    src = tmp_path / "stls"
    src.mkdir()
    for name in ("inlet", "frontBack"):
        (src / f"{name}.stl").write_text("solid\nendsolid\n")
    case = tmp_path / "case"
    case.mkdir()

    tools.import_geometry(str(case), _manifest_dict(), stl_source_dir=str(src))

    for name in ("inlet", "frontBack"):
        assert (case / "constant" / "triSurface" / f"{name}.stl").is_file()


def test_import_geometry_missing_stl_raises_before_writing_manifest(
    tmp_path: Path,
) -> None:
    """A patch whose STL is absent is an error, and no half-staged manifest is left."""
    with pytest.raises(ValueError, match="inlet"):
        tools.import_geometry(str(tmp_path), _manifest_dict())
    assert not (tmp_path / "manifest.json").exists()


def test_load_case_surfaces_dropped_configs_as_warnings(solver: Any, tmp_path: Path) -> None:
    """A present-but-invalid config is reported in warnings, not silently nulled (F5)."""
    # A transportProperties that parses as OpenFOAM but does NOT validate into
    # TransportPropertiesConfig (``nu`` is a scalar; a vector fails the schema). This
    # is the F5 case — present but schema-invalid — and it raises a catchable Python
    # error. A *syntactically* malformed dict is a different beast: OpenFOAM's
    # tokenizer calls ``FatalIOError::exit()`` at the C++ level (pybFoam exposes no
    # ``throwExceptions``), which aborts the process and no ``except`` can catch — so
    # it is out of scope for this Python-level warning path.
    (tmp_path / "constant").mkdir()
    (tmp_path / "constant" / "transportProperties").write_text(
        "transportModel   Newtonian;\nnu   ( 1 2 3 );\n"
    )

    dto = tools.load_case(solver, str(tmp_path))

    assert dto.values["transport_properties_config"] is None
    hit = next(w for w in dto.warnings if w.file == "constant/transportProperties")
    assert hit.config == "TransportPropertiesConfig" and hit.reason


def test_load_case_has_no_warnings_for_a_clean_case(solver: Any) -> None:
    """A case that spells every PIMPLE key out round-trips with an empty warnings list (F5)."""
    assert tools.load_case(solver, str(CLEAN_CASE)).warnings == []


def test_load_case_warns_about_a_slice_with_missing_keys(solver: Any) -> None:
    """pitzDaily groups ``U`` as ``"(U|nuTilda)"``, which the PIMPLE slice does not read."""
    warnings = tools.load_case(solver, str(SOURCE_CASE)).warnings

    assert [(w.file, w.config, w.reason) for w in warnings] == [
        ("system/fvSolution", "Pimple_fvSolution", "missing solvers.U, solvers.UFinal")
    ]


def test_validate_case_reads_constraint_patches_from_manifest_pre_mesh(
    solver: Any, tmp_path: Path
) -> None:
    """Before any mesh dict, a manifest's constraint patch drives the BC check (F4)."""
    tri = tmp_path / "constant" / "triSurface"
    tri.mkdir(parents=True)
    for name in ("inlet", "frontBack"):
        (tri / f"{name}.stl").write_text("solid\nendsolid\n")
    tools.import_geometry(str(tmp_path), _manifest_dict())
    # frontBack is an 'empty' constraint patch in the manifest; a mismatched BC on it
    # must be flagged even though no blockMeshDict/polyMesh exists yet.
    _write(
        tmp_path / "0" / "U",
        "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
        "boundaryField{ frontBack{ type symmetry; }\n"
        "  inlet{ type fixedValue; value uniform (1 0 0); } }\n",
    )
    report = tools.validate_case(solver, str(tmp_path))
    assert any(
        f.file == "0/U" and "frontBack" in f.message and "empty" in f.message
        for f in report.findings
    )


@pytest.mark.parametrize("verb", ["read_case", "load_case", "validate_case", "case_patches"])
def test_read_verbs_reject_an_escaping_case_dir(solver: Any, tmp_path: Path, verb: str) -> None:
    ws = Workspace.at(tmp_path)
    fn = getattr(tools, verb)  # test-only reflection over the tool surface
    args = (solver, "../escape") if verb != "case_patches" else ("../escape",)
    with pytest.raises(CaseAccessError):
        fn(*args, workspace=ws)


#: Every ``type`` string the shipped plugins answer to, per family — the same
#: inventory ``test/postprocess/test_imports.py`` pins against the registries
#: (re-stated here because ``test/mcp`` cannot import it: it is not a package).
SHIPPED_POST_TYPES: dict[str, set[str]] = {
    "Source": {"internal", "patch", "line", "plane", "isoSurface", "residuals"},
    "Node": {
        "box",
        "sphere",
        "not",
        "binary",
        "directional",
        "mag",
        "component",
        "area",
        "sample",
        "sum",
        "mean",
        "max",
        "min",
        "surfIntegrate",
        "volIntegrate",
        "rows",
        "scale",
        "print",
    },
    "TableWriter": {"csv"},
    "WriteControl": {"stepper", "timeStep", "runTime"},
}


def _post_catalog_by_key() -> dict[tuple[str, str], Any]:
    return {(entry.family, entry.type): entry for entry in tools.post_catalog()}


def _post_spec() -> dict[str, Any]:
    """One declared table: an internal source, an aggregating node, a cadence."""
    return {
        "tables": [
            {
                "name": "volume_p",
                "source": {"type": "internal", "field": "p"},
                "pipeline": [{"type": "volIntegrate", "name": "volume_p"}],
                "write_control": {"write_control_type": "timeStep", "interval": 10},
            }
        ]
    }


@pytest.mark.parametrize(("family", "shipped"), sorted(SHIPPED_POST_TYPES.items()))
def test_post_catalog_lists_every_shipped_type_of_a_family(family: str, shipped: set[str]) -> None:
    """The catalog is where a spec author learns the open mappings' ``type`` strings."""
    published = {entry.type for entry in tools.post_catalog() if entry.family == family}

    assert shipped <= published


def test_post_catalog_publishes_a_nested_region_as_an_open_object() -> None:
    """``not``/``binary`` hold another selector, and the schema has to admit one.

    The nested annotation is a bare ``Selector``, which serialises as a closed,
    property-less schema: an agent validating its payload against the catalog —
    the catalog's whole purpose — would refuse the very mapping ``save_post``
    accepts.
    """
    by_key = _post_catalog_by_key()
    region = by_key[("Node", "not")].json_schema["properties"]["region"]
    left = by_key[("Node", "binary")].json_schema["properties"]["left"]

    assert region["type"] == "object"
    assert region.get("additionalProperties") is not False
    assert left["type"] == "object"


def test_post_catalog_entry_schema_pins_its_own_type() -> None:
    by_key = _post_catalog_by_key()
    assert by_key[("Node", "volIntegrate")].json_schema["properties"]["type"]["const"] == (
        "volIntegrate"
    )
    # the WriteControl family discriminates on ``write_control_type``, not ``type``
    policy = by_key[("WriteControl", "timeStep")].json_schema["properties"]
    assert policy["write_control_type"]["const"] == "timeStep"


def test_post_catalog_marks_the_node_that_accepts_an_aggregation() -> None:
    by_key = _post_catalog_by_key()
    assert by_key[("Node", "print")].accepts_aggregated is True
    assert by_key[("Node", "sum")].accepts_aggregated is False
    # the flag is a node's; the other families carry none
    assert by_key[("TableWriter", "csv")].accepts_aggregated is None


def test_post_catalog_marks_the_source_that_aggregates_itself() -> None:
    by_key = _post_catalog_by_key()
    assert by_key[("Source", "residuals")].self_aggregating is True
    assert by_key[("Source", "internal")].self_aggregating is False
    assert by_key[("Node", "sum")].self_aggregating is None


def test_save_post_writes_a_spec_the_config_reads_back(tmp_path: Path) -> None:
    result = tools.save_post(str(tmp_path), _post_spec())

    assert result.written == ["system/postProcess.yaml"]
    assert result.tables == ["volume_p"]
    assert PostProcessConfig.load(case_dir=str(tmp_path)) == PostProcessConfig.model_validate(
        _post_spec()
    )


def test_save_post_rejects_an_unknown_node_type_naming_the_table_and_position(
    tmp_path: Path,
) -> None:
    spec = _post_spec()
    spec["tables"][0]["pipeline"].append({"type": "nope"})

    with pytest.raises(ValueError, match=r"'volume_p'.*pipeline\[1\]"):
        tools.save_post(str(tmp_path), spec)
    # the tables are resolved before anything is written
    assert not (tmp_path / "system" / "postProcess.yaml").exists()


def test_save_post_replaces_the_previous_spec(tmp_path: Path) -> None:
    """The file is written whole: a second call is the case's table set, not an addition."""
    two_tables = _post_spec()
    two_tables["tables"].append({**_post_spec()["tables"][0], "name": "volume_U"})
    tools.save_post(str(tmp_path), two_tables)

    tools.save_post(str(tmp_path), _post_spec())

    assert [t.name for t in PostProcessConfig.load(case_dir=str(tmp_path)).tables] == ["volume_p"]


def test_save_post_refuses_to_shadow_a_spec_file_the_loader_prefers(tmp_path: Path) -> None:
    """A ``.yaml`` beside a hand-authored ``.json`` would make the ``.json`` dead."""
    existing = tmp_path / "system" / "postProcess.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(_post_spec()))

    with pytest.raises(ValueError, match=r"system/postProcess\.json"):
        tools.save_post(str(tmp_path), _post_spec())
    assert not (tmp_path / "system" / "postProcess.yaml").exists()


def _preprocess_spec() -> dict[str, Any]:
    """A two-tool pipeline: build the mesh, then initialise the fields on it."""
    return {
        "tools": [
            {"tool": "blockMesh"},
            {"tool": "setFields", "depends_on": ["blockMesh"]},
        ]
    }


def _set_fields_spec() -> dict[str, Any]:
    """One default plus one box region — the declaration setFields reads."""
    return {
        "defaults": {"alpha.water": 0.0},
        "regions": [
            {
                "region": {"type": "box", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
                "values": {"alpha.water": 1.0},
            }
        ],
    }


def test_save_preprocess_writes_a_pipeline_and_a_spec_both_configs_read_back(
    tmp_path: Path,
) -> None:
    result = tools.save_preprocess(str(tmp_path), _preprocess_spec(), _set_fields_spec())

    assert result.written == ["system/preprocess.yaml", "system/setFields.yaml"]
    assert result.tools == ["blockMesh", "setFields"]
    assert PreprocessConfig.load(case_dir=str(tmp_path)) == PreprocessConfig.model_validate(
        _preprocess_spec()
    )
    assert SetFieldsConfig.load(case_dir=str(tmp_path)) == SetFieldsConfig.model_validate(
        _set_fields_spec()
    )


def test_save_preprocess_rejects_an_unknown_tool_naming_the_entry(tmp_path: Path) -> None:
    spec = _preprocess_spec()
    spec["tools"].append({"tool": "nope"})

    with pytest.raises(ValueError, match=r"tools\[2\] 'nope'"):
        tools.save_preprocess(str(tmp_path), spec)
    # the pipeline is resolved before anything is written
    assert not (tmp_path / "system" / "preprocess.yaml").exists()


def test_save_preprocess_rejects_an_invalid_step_option_naming_the_entry(tmp_path: Path) -> None:
    spec = _preprocess_spec()
    spec["tools"][0]["verbose"] = "maybe"

    with pytest.raises(ValueError, match=r"tools\[0\] 'blockMesh'"):
        tools.save_preprocess(str(tmp_path), spec)
    assert not (tmp_path / "system" / "preprocess.yaml").exists()


def test_save_preprocess_rejects_an_unknown_region_type_naming_the_position(
    tmp_path: Path,
) -> None:
    fields = _set_fields_spec()
    fields["regions"].append({"region": {"type": "nope"}, "values": {"alpha.water": 1.0}})

    with pytest.raises(ValueError, match=r"regions\[1\]"):
        tools.save_preprocess(str(tmp_path), _preprocess_spec(), fields)
    # the regions are resolved before anything is written
    assert not (tmp_path / "system" / "setFields.yaml").exists()


def test_save_preprocess_rejects_set_fields_the_pipeline_never_reads(tmp_path: Path) -> None:
    """A spec file no entry runs setFields for is a silent mistake, not a no-op."""
    pipeline = {"tools": [{"tool": "blockMesh"}]}

    with pytest.raises(ValueError, match="setFields"):
        tools.save_preprocess(str(tmp_path), pipeline, _set_fields_spec())
    assert not (tmp_path / "system" / "setFields.yaml").exists()


def test_save_preprocess_without_set_fields_leaves_an_existing_declaration_alone(
    tmp_path: Path,
) -> None:
    write_configs([SetFieldsConfig.model_validate(_set_fields_spec())], case_dir=str(tmp_path))
    before = (tmp_path / "system" / "setFields.yaml").read_text()

    result = tools.save_preprocess(str(tmp_path), _preprocess_spec())

    assert result.written == ["system/preprocess.yaml"]
    assert (tmp_path / "system" / "setFields.yaml").read_text() == before


@pytest.mark.parametrize("depends_on", [["nosuch"], "blockMesh"], ids=["unknown", "bare_string"])
def test_save_preprocess_rejects_a_depends_on_the_pipeline_cannot_satisfy(
    tmp_path: Path, depends_on: Any
) -> None:
    """Neither reaches the run as itself: an unknown name and a string that reads as
    a list of characters both surface there as "must have exactly one sink tool"."""
    spec = _preprocess_spec()
    spec["tools"][1]["depends_on"] = depends_on

    with pytest.raises(ValueError, match=r"tools\[1\] 'setFields'.*depends_on"):
        tools.save_preprocess(str(tmp_path), spec)
    assert not (tmp_path / "system" / "preprocess.yaml").exists()


def test_save_preprocess_replaces_the_previous_pipeline(tmp_path: Path) -> None:
    tools.save_preprocess(str(tmp_path), _preprocess_spec())

    result = tools.save_preprocess(str(tmp_path), {"tools": [{"tool": "blockMesh"}]})

    assert result.tools == ["blockMesh"]
    assert PreprocessConfig.load(case_dir=str(tmp_path)) == PreprocessConfig.model_validate(
        {"tools": [{"tool": "blockMesh"}]}
    )


def test_save_preprocess_refuses_to_shadow_a_set_fields_file_the_loader_prefers(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "system" / "setFields.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(_set_fields_spec()))

    with pytest.raises(ValueError, match=r"system/setFields\.json"):
        tools.save_preprocess(str(tmp_path), _preprocess_spec(), _set_fields_spec())
    assert not (tmp_path / "system" / "setFields.yaml").exists()
    # nothing at all is written: the pipeline shares the call
    assert not (tmp_path / "system" / "preprocess.yaml").exists()
