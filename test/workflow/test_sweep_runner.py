# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Per-case clone + config apply behind the generated Snakefile's setup rule."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.mcp import tools  # noqa: E402
from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.solver.incompressibleFluid.models.boussinesq import BoussinesqConfig  # noqa: E402
from neofoam.ui.scaffold import scaffold_runnable_case  # noqa: E402
from neofoam.viscosity.config import TransportPropertiesConfig  # noqa: E402
from neofoam.workflow.sweep_runner import (  # noqa: E402
    apply_configs,
    clone_case,
    config_classes_by_name,
    main,
    setup_case,
)

_CONTROL = {
    "application": "pimpleFoam",
    "endTime": 1.0,
    "deltaT": 0.001,
    "writeControl": "timeStep",
    "writeInterval": 1,
}


def _make_base(
    tmp_path: Path, extra: dict[str, dict[str, object]] | None = None
) -> Path:
    base = tmp_path / "base"
    solver = resolve_solver("incompressibleFluid")
    spec: dict[str, object] = {
        "transport_properties_config": {"transportModel": "Newtonian", "nu": 1e-5},
        "control_dict_config": dict(_CONTROL),
        **(extra or {}),
    }
    tools.save_case(solver, spec, str(base))
    scaffold_runnable_case(base)
    return base


def test_config_classes_by_name_round_trips_snake_names() -> None:
    classes = config_classes_by_name(resolve_solver("incompressibleFluid"))
    assert classes["transport_properties_config"] is TransportPropertiesConfig
    assert "control_dict_config" in classes


def test_clone_case_drops_artifacts_keeps_allrun(tmp_path: Path) -> None:
    base = _make_base(tmp_path)
    (base / "log.solver").write_text("old log")
    (base / "case.foam").write_text("")
    (base / "0.5").mkdir()
    (base / "0.5" / "U").write_text("stale result")
    (base / "postProcessing").mkdir()

    dest = tmp_path / "cases" / "clone"
    clone_case(base, dest)

    assert (dest / "constant" / "transportProperties").is_file()
    assert os.access(dest / "Allrun", os.X_OK)
    assert not (dest / "log.solver").exists()
    assert not (dest / "case.foam").exists()
    assert not (dest / "0.5").exists()
    assert not (dest / "postProcessing").exists()


def test_setup_case_applies_swept_config(tmp_path: Path) -> None:
    base = _make_base(tmp_path)
    config_json = tmp_path / "setup.json"
    config_json.write_text(
        json.dumps(
            {"transport_properties_config": {"transportModel": "Newtonian", "nu": 2e-5}}
        )
    )
    case = tmp_path / "cases" / "nu2"
    stamp = case / ".applied.json"

    rc = main(
        [
            "setup",
            "--solver",
            "incompressibleFluid",
            "--base",
            str(base),
            "--case",
            str(case),
            "--config",
            str(config_json),
            "--stamp",
            str(stamp),
        ]
    )
    assert rc == 0
    assert TransportPropertiesConfig.load(case_dir=case).nu == pytest.approx(2e-5)
    # The unswept controlDict is untouched — byte-identical to the base.
    assert (case / "system" / "controlDict").read_bytes() == (
        base / "system" / "controlDict"
    ).read_bytes()
    assert json.loads(stamp.read_text())["transport_properties_config"]["nu"] == 2e-5
    # Base keeps its own value.
    assert TransportPropertiesConfig.load(case_dir=base).nu == pytest.approx(1e-5)


def test_apply_configs_preserves_co_owned_keys(tmp_path: Path) -> None:
    # transportProperties is co-owned: sweeping TransportPropertiesConfig must
    # not drop the Boussinesq keys the base case carries in the same file.
    base = _make_base(
        tmp_path,
        extra={
            "boussinesq_config": {"beta": 3e-3, "TRef": 300.0, "Pr": 0.7, "Prt": 0.85}
        },
    )
    case = tmp_path / "cases" / "sweep"
    written = setup_case(
        "incompressibleFluid",
        base,
        case,
        _write_json(
            tmp_path,
            {
                "transport_properties_config": {
                    "transportModel": "Newtonian",
                    "nu": 2e-5,
                }
            },
        ),
    )
    assert "constant/transportProperties" in written
    assert TransportPropertiesConfig.load(case_dir=case).nu == pytest.approx(2e-5)
    survived = BoussinesqConfig.load(case_dir=case)
    assert survived.beta == pytest.approx(3e-3)
    assert survived.TRef == pytest.approx(300.0)


def test_apply_configs_rejects_unknown_or_invalid(tmp_path: Path) -> None:
    base = _make_base(tmp_path)
    solver = resolve_solver("incompressibleFluid")
    case = tmp_path / "cases" / "x"
    clone_case(base, case)

    with pytest.raises(ValueError, match="unknown config 'not_a_config'"):
        apply_configs(solver, case, {"not_a_config": {}})
    with pytest.raises(ValueError, match="failed validation"):
        apply_configs(
            solver, case, {"transport_properties_config": {"nu": "not-a-number"}}
        )


def _write_json(tmp_path: Path, payload: dict[str, dict[str, object]]) -> Path:
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(payload))
    return path


# --- shared keyed mesh: mesh-setup / tool / setup --mesh-src ------------------

_MANIFEST = Path(__file__).parent / "cases" / "tube_bank_manifest.json"
_TRI = Path(__file__).parent / "cases" / "tube_bank" / "constant" / "triSurface"

# Minimal system dicts: constructing an fvMesh requires fvSchemes/fvSolution to
# exist; blockMesh itself only reads blockMeshDict (same trick as the
# test/tools/_blockmesh_probe.py subprocess helper).
_SCHEMES = (
    "FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\n"
    "ddtSchemes{default steadyState;}gradSchemes{default Gauss linear;}"
    "divSchemes{default none;}laplacianSchemes{default Gauss linear corrected;}"
    "interpolationSchemes{default linear;}snGradSchemes{default corrected;}"
)
_SOLUTION = "FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\n"


def _make_meshable_base(tmp_path: Path) -> Path:
    """A base case carrying everything the mesh mini-case stages."""
    import shutil

    from neofoam.framework.tools import PreprocessConfig
    from neofoam.io import write_configs
    from neofoam.workflow.mesh_inputs import block_mesh_dict, snappy_dict
    from neofoam.workflow.patch_set import PatchSet

    base = _make_base(tmp_path)
    patch_set = PatchSet.load(_MANIFEST)
    write_configs([block_mesh_dict(patch_set), snappy_dict(patch_set)], base)
    PreprocessConfig(
        tools=[
            {"tool": "blockMesh", "verbose": False},
            {"tool": "snappyHexMesh", "depends_on": ["blockMesh"]},
            {"tool": "checkMesh", "depends_on": ["snappyHexMesh"]},
        ]
    ).save(case_dir=base)
    (base / "system" / "fvSchemes").write_text(_SCHEMES)
    (base / "system" / "fvSolution").write_text(_SOLUTION)
    shutil.copytree(_TRI, base / "constant" / "triSurface")
    return base


def test_setup_mesh_case_stages_and_applies(tmp_path: Path) -> None:
    from neofoam.tools.block_mesh import BlockMeshDictConfig

    base = _make_meshable_base(tmp_path)
    mesh_dir = tmp_path / "meshes" / "coarse"
    # Stale mesh from a previous variant definition must be wiped on re-stage.
    (mesh_dir / "constant" / "polyMesh").mkdir(parents=True)
    (mesh_dir / "constant" / "polyMesh" / "points").write_text("stale")

    config = tmp_path / "coarse.json"
    config.write_text(json.dumps({"block_mesh_dict_config": {"scale": 0.5}}))
    stamp = mesh_dir / ".staged.json"
    rc = main(
        [
            "mesh-setup",
            "--solver",
            "incompressibleFluid",
            "--base",
            str(base),
            "--case",
            str(mesh_dir),
            "--config",
            str(config),
            "--stamp",
            str(stamp),
        ]
    )
    assert rc == 0
    for rel in (
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/blockMeshDict",
        "system/snappyHexMeshDict",
        "constant/triSurface/tubes.stl",
    ):
        assert (mesh_dir / rel).is_file(), f"missing {rel}"
    assert not (mesh_dir / "system" / "preprocess.yaml").exists()
    assert not (mesh_dir / "constant" / "polyMesh").exists()
    assert stamp.is_file()
    # The variant payload landed in the mini-case; the base is untouched.
    assert BlockMeshDictConfig.load(case_dir=mesh_dir).scale == pytest.approx(0.5)
    assert BlockMeshDictConfig.load(case_dir=base).scale != pytest.approx(0.5)


def test_setup_mesh_case_implicit_variant_applies_nothing(tmp_path: Path) -> None:
    from neofoam.workflow.sweep_runner import setup_mesh_case

    base = _make_meshable_base(tmp_path)
    mesh_dir = tmp_path / "meshes" / "base"
    config = tmp_path / "base.json"
    config.write_text("{}\n")
    written = setup_mesh_case("incompressibleFluid", base, mesh_dir, config)
    assert written == []
    assert (mesh_dir / "system" / "blockMeshDict").read_bytes() == (
        base / "system" / "blockMeshDict"
    ).read_bytes()


def test_setup_mesh_case_requires_control_dict(tmp_path: Path) -> None:
    from neofoam.workflow.sweep_runner import setup_mesh_case

    bare = tmp_path / "bare"
    bare.mkdir()
    config = tmp_path / "c.json"
    config.write_text("{}")
    with pytest.raises(FileNotFoundError, match="no system/controlDict"):
        setup_mesh_case("incompressibleFluid", bare, tmp_path / "meshes" / "x", config)


def test_run_tool_unknown_tool_errors(tmp_path: Path) -> None:
    from neofoam.workflow.sweep_runner import run_tool

    base = _make_meshable_base(tmp_path)
    with pytest.raises(ValueError, match="'fluxMagic' is not in the base case's"):
        run_tool("fluxMagic", base, tmp_path / "meshes" / "x")


def test_setup_case_with_mesh_src_copies_polymesh(tmp_path: Path) -> None:
    base = _make_meshable_base(tmp_path)
    mesh_src = tmp_path / "meshes" / "base"
    (mesh_src / "constant" / "polyMesh").mkdir(parents=True)
    (mesh_src / "constant" / "polyMesh" / "points").write_text("mesh points")

    case = tmp_path / "cases" / "nu2"
    written = setup_case(
        "incompressibleFluid",
        base,
        case,
        _write_json(
            tmp_path,
            {
                "transport_properties_config": {
                    "transportModel": "Newtonian",
                    "nu": 2e-5,
                }
            },
        ),
        mesh_src=mesh_src,
    )
    assert written
    assert (case / "constant" / "polyMesh" / "points").read_text() == "mesh points"
    # The clone must never re-mesh itself with un-swept dicts.
    assert not (case / "system" / "preprocess.yaml").exists()


def test_setup_case_missing_mesh_src_errors(tmp_path: Path) -> None:
    base = _make_base(tmp_path)
    with pytest.raises(FileNotFoundError, match="run the mesh rules first"):
        setup_case(
            "incompressibleFluid",
            base,
            tmp_path / "cases" / "x",
            _write_json(tmp_path, {}),
            mesh_src=tmp_path / "meshes" / "nope",
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("WM_PROJECT_DIR"), reason="needs a sourced OpenFOAM"
)
def test_run_tool_blockmesh_persists_polymesh(tmp_path: Path) -> None:
    """blockMesh via the single-tool slice leaves constant/polyMesh on disk.

    Runs in a subprocess: in-process pybFoam meshing can crash at GC teardown
    (see test/tools/_blockmesh_probe.py), and each Snakemake job is its own
    process anyway — this mirrors production.
    """
    import subprocess
    import sys

    from neofoam.workflow.sweep_runner import setup_mesh_case

    base = _make_meshable_base(tmp_path)
    mesh_dir = tmp_path / "meshes" / "base"
    setup_mesh_case("incompressibleFluid", base, mesh_dir, _write_json(tmp_path, {}))

    stamp = mesh_dir / ".blockMesh.done"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "neofoam.workflow.sweep_runner",
            "tool",
            "--tool",
            "blockMesh",
            "--base",
            str(base),
            "--case",
            str(mesh_dir),
            "--stamp",
            str(stamp),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert (mesh_dir / "constant" / "polyMesh" / "points").is_file()
    assert (mesh_dir / "system" / "preprocess.yaml").read_text().count("tool:") == 1
    assert json.loads(stamp.read_text())["tool"] == "blockMesh"
