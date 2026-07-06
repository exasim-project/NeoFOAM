# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Per-case / per-variant workers behind the generated sweep workflows.

The packaged Snakemake rules (:mod:`neofoam.workflow.rules`) shell out to::

    python -m neofoam.workflow.sweep_runner mesh-setup ...   # stage meshes/<variant>
    python -m neofoam.workflow.sweep_runner tool ...         # run ONE mesh tool there
    python -m neofoam.workflow.sweep_runner setup ...        # clone base + apply configs

``setup`` clones the wizard-saved base case and re-writes the swept config
files from the case's materialized JSON payloads (each validated — and thereby
coerced — by its config class; co-owners of a target file are reloaded from the
clone first so re-writing one config never drops keys another config owns).
``mesh-setup``/``tool`` implement the shared keyed mesh: a mesh variant is a
mini-case under ``meshes/<variant>/`` staged from the base, meshed once by
per-tool rules (each writes a single-tool ``system/preprocess.yaml`` slice and
runs the in-process preprocess DAG), then copied into every case that selects
it.

Unlike :mod:`neofoam.workflow.paramspace` this module runs inside a full
neofoam environment (it needs pybFoam for the OpenFOAM dictionary IO).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neofoam.framework.solver.configurations import _snake_case, configurations
from neofoam.framework.tools import PreprocessConfig
from neofoam.io import BaseConfig, write_configs
from neofoam.mcp.registry import resolve_solver

__all__ = [
    "apply_configs",
    "clone_case",
    "config_classes_by_name",
    "main",
    "run_tool",
    "setup_case",
    "setup_mesh_case",
]

#: Case content that must not leak from the base into a fresh clone.
_CLONE_IGNORES = ("postProcessing", "processor*", "log.*", "*.foam")

#: What a mesh mini-case needs from the base: Time wants controlDict, building
#: an fvMesh wants fvSchemes/fvSolution, the tools read their dicts, snappy
#: reads the STLs. ``preprocess.yaml`` is deliberately NOT staged — each
#: per-tool rule writes its own single-tool slice.
_MESH_STAGE_FILES = (
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
    "system/blockMeshDict",
    "system/snappyHexMeshDict",
)
_MESH_STAGE_TREES = ("constant/triSurface",)


def config_classes_by_name(solver: Any) -> dict[str, type[BaseConfig]]:
    """The solver's config classes keyed by their snake-case name.

    The snake name (``transport_properties_config``) is the sweep-dimension
    key used in ``params.yaml`` / ``configs/{case}/setup.json``.
    """
    return {_snake_case(cls.__name__): cls for cls in configurations(solver).classes}


def _is_time_dir(name: str) -> bool:
    return name.replace(".", "", 1).isdigit()


def clone_case(base: Path, dest: Path) -> None:
    """Copy the base case to ``dest``, dropping run artifacts and results.

    Removes an existing ``dest`` first (a re-run of ``setup`` starts clean).
    ``copytree`` preserves permissions, so the scaffolded ``Allrun``/``Allclean``
    stay executable. Time directories other than ``0`` (previous results) are
    dropped.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base, dest, ignore=shutil.ignore_patterns(*_CLONE_IGNORES))
    for child in dest.iterdir():
        if child.is_dir() and _is_time_dir(child.name) and float(child.name) > 0:
            shutil.rmtree(child)


def apply_configs(
    solver: Any,
    case_dir: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Validate and write the swept config payloads into ``case_dir``.

    Co-owners of a swept target file (e.g. ``BoussinesqConfig`` sharing
    ``constant/transportProperties`` with ``TransportPropertiesConfig``) are
    reloaded from the case and written together with the swept configs — swept
    last, so their keys win — because the OpenFOAM writer clears each file
    before rewriting it.

    Returns:
        The relative paths of the files written.

    Raises:
        ValueError: On a payload naming an unknown config or failing validation.
    """
    classes = config_classes_by_name(solver)

    swept: list[BaseConfig] = []
    target_files: set[str] = set()
    for name, payload in payloads.items():
        cls = classes.get(name)
        if cls is None:
            msg = f"unknown config '{name}' (known: {sorted(classes)})"
            raise ValueError(msg)
        if cls.io_config is None:
            msg = f"config '{name}' has no file binding — it cannot be swept"
            raise ValueError(msg)
        try:
            swept.append(cls.model_validate(dict(payload)))
        except Exception as e:
            msg = f"payload for '{name}' failed validation against {cls.__name__}: {e}"
            raise ValueError(msg) from e
        target_files.add(cls.io_config.file)

    co_owners: list[BaseConfig] = []
    for other_name, cls in classes.items():
        io = cls.io_config
        if other_name in payloads or io is None or io.file not in target_files:
            continue
        if not (case_dir / io.file).exists():
            continue
        try:
            co_owners.append(cls.load(case_dir=case_dir))
        except Exception:
            # File present but not (or no longer) carrying this config's slice —
            # same tolerant read as load_case_from_disk.
            continue

    report = write_configs([*co_owners, *swept], case_dir)
    return sorted(report)


def _load_payloads(config_json: Path) -> dict[str, Any]:
    payloads = json.loads(Path(config_json).read_text())
    if not isinstance(payloads, dict):
        msg = f"config file {config_json}: expected a mapping of config payloads"
        raise ValueError(msg)
    return payloads


def _write_stamp(stamp: Path | None, payload: Mapping[str, Any]) -> None:
    if stamp is None:
        return
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def setup_case(
    solver_name: str,
    base: Path,
    case_dir: Path,
    config_json: Path,
    stamp: Path | None = None,
    mesh_src: Path | None = None,
) -> list[str]:
    """Clone ``base`` into ``case_dir`` and apply the materialized payloads.

    Args:
        solver_name: Registered solver name (resolves the config classes).
        base: The wizard-saved base case.
        case_dir: The per-case clone to (re)create.
        config_json: The case's ``configs/{case}/setup.json``.
        stamp: Optional stamp file written last (the Snakemake rule output);
            holds the applied payloads with a fresh mtime.
        mesh_src: A meshed variant dir (``meshes/<variant>``) whose
            ``constant/polyMesh`` is copied into the clone. The clone's
            ``system/preprocess.yaml`` is then removed so nothing — not even a
            manual ``./Allrun`` — re-meshes the case with un-swept dicts.

    Returns:
        The relative paths of the config files written into the clone.
    """
    solver = resolve_solver(solver_name)
    payloads = _load_payloads(config_json)

    clone_case(base, case_dir)
    written = apply_configs(solver, case_dir, payloads)

    if mesh_src is not None:
        poly_src = Path(mesh_src) / "constant" / "polyMesh"
        if not poly_src.is_dir():
            msg = f"mesh source {poly_src} does not exist — run the mesh rules first"
            raise FileNotFoundError(msg)
        poly_dst = case_dir / "constant" / "polyMesh"
        if poly_dst.exists():
            shutil.rmtree(poly_dst)
        shutil.copytree(poly_src, poly_dst)
        (case_dir / "system" / "preprocess.yaml").unlink(missing_ok=True)

    _write_stamp(stamp, payloads)
    return written


def setup_mesh_case(
    solver_name: str,
    base: Path,
    mesh_dir: Path,
    config_json: Path,
    stamp: Path | None = None,
) -> list[str]:
    """Stage one mesh variant's mini-case under ``mesh_dir`` (``meshes/<variant>``).

    Copies the mesh-relevant inputs from the base case (dicts, STLs — see
    ``_MESH_STAGE_FILES``/``_MESH_STAGE_TREES``), wipes any stale
    ``constant/polyMesh`` (a re-stage means the variant changed: the mesh must
    be rebuilt from scratch), and applies the variant's config payloads
    (config-name-keyed, e.g. ``{"block_mesh_dict_config": {...}}``; ``{}`` for
    the implicit ``base`` variant).

    Returns:
        The relative paths of the config files written (empty for ``{}``).
    """
    payloads = _load_payloads(config_json)

    control = base / "system" / "controlDict"
    if not control.is_file():
        msg = (
            f"base case {base} has no system/controlDict — a mesh mini-case "
            f"cannot construct an OpenFOAM Time without one (save the case first)"
        )
        raise FileNotFoundError(msg)

    for rel in _MESH_STAGE_FILES:
        src = base / rel
        if not src.is_file():
            continue
        dst = mesh_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in _MESH_STAGE_TREES:
        src = base / rel
        if not src.is_dir():
            continue
        dst = mesh_dir / rel
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    stale = mesh_dir / "constant" / "polyMesh"
    if stale.exists():
        shutil.rmtree(stale)

    written: list[str] = []
    if payloads:
        written = apply_configs(resolve_solver(solver_name), mesh_dir, payloads)

    _write_stamp(stamp, payloads)
    return written


def run_tool(
    tool_name: str,
    base: Path,
    case_dir: Path,
    stamp: Path | None = None,
) -> None:
    """Run ONE preprocess tool in ``case_dir`` via a single-tool pipeline slice.

    The tool's options come from the base case's ``system/preprocess.yaml``
    entry (minus ``depends_on`` — chaining is Snakemake's job here); the slice
    is written to ``case_dir/system/preprocess.yaml`` and executed in-process
    from within ``case_dir`` (tool dict paths are CWD-relative). A
    mesh-consuming tool without a predecessor (snappyHexMesh, checkMesh)
    resumes from the ``constant/polyMesh`` already on disk.
    """
    from neofoam.tools.run import run_preprocess  # lazy: binds pybFoam meshing
    from neofoam.workflow.paramspace import write_if_changed

    base = Path(base).resolve()
    case_dir = Path(case_dir).resolve()
    stamp = Path(stamp).resolve() if stamp is not None else None

    cfg = PreprocessConfig.load(case_dir=base)
    entry = next((dict(e) for e in cfg.tools if e.get("tool") == tool_name), None)
    if entry is None:
        available = sorted(str(e.get("tool")) for e in cfg.tools)
        msg = (
            f"tool '{tool_name}' is not in the base case's preprocess pipeline "
            f"({base / 'system' / 'preprocess.yaml'}; available: {available})"
        )
        raise ValueError(msg)
    entry.pop("depends_on", None)

    import yaml

    slice_text = yaml.safe_dump({"tools": [entry]}, sort_keys=True)
    write_if_changed(case_dir / "system" / "preprocess.yaml", slice_text)

    cwd = Path.cwd()
    os.chdir(case_dir)
    try:
        run_preprocess([f"neofoam-sweep-{tool_name}", "-case", "."])
    finally:
        os.chdir(cwd)

    _write_stamp(stamp, {"tool": tool_name, "options": entry})


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by generated Snakefiles."""
    parser = argparse.ArgumentParser(
        prog="python -m neofoam.workflow.sweep_runner",
        description="Per-case / per-mesh-variant workers for NeoFOAM sweep workflows.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser(
        "setup", help="clone the base case and apply the case's configs"
    )
    setup.add_argument("--solver", required=True, help="registered solver name")
    setup.add_argument("--base", required=True, type=Path, help="the saved base case")
    setup.add_argument(
        "--case", required=True, type=Path, help="the per-case clone to create"
    )
    setup.add_argument(
        "--config", required=True, type=Path, help="configs/<case>/setup.json"
    )
    setup.add_argument(
        "--stamp", type=Path, default=None, help="stamp file to write on success"
    )
    setup.add_argument(
        "--mesh-src",
        type=Path,
        default=None,
        help="meshed variant dir whose constant/polyMesh is copied into the clone",
    )

    mesh_setup = sub.add_parser(
        "mesh-setup", help="stage one mesh variant's mini-case under meshes/<variant>"
    )
    mesh_setup.add_argument("--solver", required=True, help="registered solver name")
    mesh_setup.add_argument(
        "--base", required=True, type=Path, help="the saved base case"
    )
    mesh_setup.add_argument(
        "--case", required=True, type=Path, help="the meshes/<variant> dir to stage"
    )
    mesh_setup.add_argument(
        "--config", required=True, type=Path, help="configs/mesh/<variant>.json"
    )
    mesh_setup.add_argument(
        "--stamp", type=Path, default=None, help="stamp file to write on success"
    )

    tool = sub.add_parser(
        "tool", help="run ONE preprocess tool in a staged mesh variant dir"
    )
    tool.add_argument("--tool", required=True, help="tool name (e.g. blockMesh)")
    tool.add_argument(
        "--base",
        required=True,
        type=Path,
        help="the base case whose preprocess.yaml holds the tool's options",
    )
    tool.add_argument(
        "--case", required=True, type=Path, help="the meshes/<variant> dir to run in"
    )
    tool.add_argument(
        "--stamp", type=Path, default=None, help="stamp file to write on success"
    )

    args = parser.parse_args(argv)
    if args.command == "setup":
        written = setup_case(
            args.solver, args.base, args.case, args.config, args.stamp, args.mesh_src
        )
    elif args.command == "mesh-setup":
        written = setup_mesh_case(
            args.solver, args.base, args.case, args.config, args.stamp
        )
    else:
        run_tool(args.tool, args.base, args.case, args.stamp)
        written = []
    for path in written:
        print(f"wrote {args.case / path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
