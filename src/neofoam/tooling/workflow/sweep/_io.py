# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Export a sweep canvas to a runnable workflow directory, and read it back.

:func:`export_sweep` validates every variant, builds the cross product and writes
the workflow directory (``sweep.csv``, ``params.yaml``, ``Snakefile``, the
materialized ``configs/``) plus a ``sweep.meta.json`` sidecar holding the facts a
reader needs — solver name, base case, enabled rules, CAD model. :func:`load_sweep`
is the inverse: it reads ``params.yaml`` (the full ``{dim: {variant: payload}}``)
and the sidecar. The sidecar is the read-back contract, so ``load_sweep`` never
parses the generated ``Snakefile`` — the executable workflow and its metadata are
kept in separate files.

This module has no trame imports so it can be exercised headless.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from neofoam.tooling.workflow.paramspace import (
    YamlParamSpace,
    read_params_yaml,
    write_if_changed,
    write_params_yaml,
    write_sweep_csv,
)
from neofoam.tooling.workflow.rules import (
    CAD_DIM,
    DEFAULT_ENABLED,
    MESH_DIM,
    RuleRegistry,
    default_registry,
)
from neofoam.tooling.workflow.sweep._canvas import SETUP_RULE
from neofoam.tooling.workflow.sweep._codegen import (
    _check_composite_mesh_names,
    sweep_snakefile,
)
from neofoam.tooling.workflow.sweep._validate import (
    validate_cad_dimension,
    validate_dimensions,
    validate_mesh_dimension,
)

#: Rule that regenerates the STLs when a CAD axis is swept (opt-in, like post).
CAD_RULE = "cad_geometry"

#: Sidecar holding the read-back metadata (the inverse of the Snakefile header).
SWEEP_META_FILE = "sweep.meta.json"


def merge_cad(
    dimensions: Mapping[str, Mapping[str, Any]],
    cad: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Fold the CAD axis' variants into the dimension map; return (dims, cad_model).

    A CAD axis exists only once it carries variants — an empty/absent ``cad`` dict
    leaves the dimensions untouched and returns ``cad_model=None`` (a mesh-only
    sweep). Shared by :func:`export_sweep` and :class:`~…sweep.Sweep` so both derive
    the same effective dimensions and model path.
    """
    dims = {d: dict(v) for d, v in dimensions.items()}
    info = (cad or {}).get(CAD_DIM) or {}
    variants = info.get("variants")
    if variants:
        dims[CAD_DIM] = {k: dict(v) for k, v in variants.items()}
    cad_model = str(info["model"]) if CAD_DIM in dims else None
    return dims, cad_model


def plan_enabled_with_cad(
    enabled: Sequence[str] | None, *, has_cad: bool
) -> Sequence[str] | None:
    """Add the opt-in ``cad_geometry`` rule to *enabled* when a CAD axis is present."""
    if not has_cad:
        return enabled
    base = list(DEFAULT_ENABLED if enabled is None else enabled)
    if CAD_RULE not in base:
        base.append(CAD_RULE)
    return base


def cross_product(
    dimensions: Mapping[str, Mapping[str, Any]],
    case_col: str = "case",
) -> list[dict[str, str]]:
    """The full cross product of variant names per dimension (sweep.csv rows).

    Case names join the variant names in sorted-dimension order.
    """
    dim_order = sorted(dimensions)
    names_per_dim = [sorted(dimensions[dim]) for dim in dim_order]
    rows: list[dict[str, str]] = []
    seen: dict[str, tuple[str, ...]] = {}
    for combo in itertools.product(*names_per_dim):
        case = "_".join(combo) or "default"
        if case in seen:
            msg = (
                f"case-name collision: '{case}' produced by more than one variant "
                f"combination ({seen[case]} and {combo}) — rename a variant to "
                f"avoid '_' ambiguity"
            )
            raise ValueError(msg)
        seen[case] = combo
        row = {case_col: case}
        row.update(zip(dim_order, combo))
        rows.append(row)
    return rows


@dataclass(frozen=True)
class SweepExport:
    """The files written by :func:`export_sweep`."""

    out_dir: Path
    sweep_csv: Path
    params_yaml: Path
    snakefile: Path
    configs: list[Path]


def export_sweep(
    out_dir: str | Path,
    *,
    solver_name: str,
    base_case: str | Path,
    dimensions: Mapping[str, dict[str, dict[str, Any]]],
    classes: Mapping[str, type[BaseModel]],
    cad: Mapping[str, Mapping[str, Any]] | None = None,
    registry: RuleRegistry | None = None,
    enabled: Sequence[str] | None = None,
) -> SweepExport:
    """Write the runnable workflow directory for the current canvas.

    Validates every variant (the ``mesh`` dimension via
    :func:`validate_mesh_dimension`, the ``cad`` dimension via
    :func:`validate_cad_dimension`, everything else against its config class),
    builds the cross product and writes ``sweep.csv``, ``params.yaml``, the
    ``Snakefile``, the ``sweep.meta.json`` sidecar and the materialized configs
    into ``out_dir`` — per-case ``configs/{case}/setup.json`` (mesh + cad
    excluded: they apply in the mesh mini-case) plus per-variant
    ``configs/mesh/{variant}.json`` (an implicit empty ``base`` variant when the
    sweep has no mesh dimension) and, when a CAD axis is present,
    ``configs/cad/{variant}.json``. The directory is immediately runnable with
    ``snakemake``.

    Args:
        cad: The CAD axis as ``{"cad": {"model": path, "variants": {name:
            params}}}`` (the :meth:`SweepModel.cad_dimensions` shape). Empty or
            ``None`` keeps today's mesh-only behaviour. When given, the
            ``cad_geometry`` rule is enabled and the model path is threaded into
            the Snakefile.
    """
    # A CAD axis exists only once it carries variants — an empty cad dict emits
    # no CAD_MODEL / cad_axis / configs/cad and behaves like a mesh-only sweep.
    dimensions, cad_model = merge_cad(dimensions, cad)
    has_cad = CAD_DIM in dimensions
    if has_cad:
        validate_cad_dimension(dimensions[CAD_DIM])

    case_dims = {d: v for d, v in dimensions.items() if d not in (MESH_DIM, CAD_DIM)}
    validate_dimensions(case_dims, classes)
    if MESH_DIM in dimensions:
        validate_mesh_dimension(dimensions[MESH_DIM], classes)
    if not dimensions:
        msg = "no sweep dimensions on the canvas — add at least one config dimension"
        raise ValueError(msg)
    # Build the sweep rows before creating the output directory so a case-name
    # collision aborts with nothing written to disk (no partial workflow dir).
    rows = cross_product(dimensions)
    if has_cad:
        _check_composite_mesh_names(dimensions)

    # The CAD chain is opt-in (like post): enable it only when a CAD axis is on
    # the canvas so mesh-only sweeps stay untouched.
    plan_enabled = plan_enabled_with_cad(enabled, has_cad=has_cad)

    reg = registry or default_registry()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sweep_path = out / "sweep.csv"
    params_path = out / "params.yaml"
    snakefile_path = out / "Snakefile"

    write_sweep_csv(sweep_path, rows)
    write_params_yaml(params_path, {d: dict(v) for d, v in dimensions.items()})
    write_if_changed(
        snakefile_path,
        sweep_snakefile(
            solver_name,
            base_case,
            sorted(dimensions),
            cad_model=cad_model,
            registry=reg,
            enabled=plan_enabled,
        ),
    )
    _write_sweep_meta(
        out / SWEEP_META_FILE,
        solver_name=solver_name,
        base_case=str(Path(base_case).resolve()),
        enabled=list(reg.plan(plan_enabled).names),
        cad_model=cad_model or "",
    )

    space = YamlParamSpace(sweep_path, params_path)
    setup_dims = [d for d in space.dims if d not in (MESH_DIM, CAD_DIM)]
    configs = space.materialize({SETUP_RULE: setup_dims}, out_dir=out / "configs")
    if MESH_DIM in dimensions:
        configs += space.materialize_dims([MESH_DIM], out_dir=out / "configs")
    else:
        implicit = out / "configs" / MESH_DIM / "base.json"
        if write_if_changed(implicit, "{}\n"):
            configs.append(implicit)
    if has_cad:
        configs += space.materialize_dims([CAD_DIM], out_dir=out / "configs")

    return SweepExport(
        out_dir=out,
        sweep_csv=sweep_path,
        params_yaml=params_path,
        snakefile=snakefile_path,
        configs=configs,
    )


@dataclass(frozen=True)
class LoadedSweep:
    """A sweep read back from an exported directory (inverse of export)."""

    solver_name: str
    base_case: str
    dimensions: dict[str, dict[str, dict[str, Any]]]
    enabled: list[str]
    cad_model: str = ""


def load_sweep(
    out_dir: str | Path, *, registry: RuleRegistry | None = None
) -> LoadedSweep:
    """Read an exported sweep directory back into its canvas definition.

    The inverse of :func:`export_sweep`: ``params.yaml`` holds the full
    ``{dimension: {variant: payload}}`` (variants keep their complete payloads),
    and the ``sweep.meta.json`` sidecar names the solver, the base case, the
    enabled rules and the CAD model. A sweep exported without a mesh dimension
    has no ``mesh`` key (the implicit ``base`` mesh variant is materialized
    separately, not a canvas dimension), so it round-trips to the same canvas.

    The ``registry`` argument is accepted for backward compatibility and is
    unused — the sidecar records the enabled rules directly, so no rule-name
    lookup is needed.

    Raises:
        ValueError: If ``out_dir`` has no ``params.yaml`` (not an exported
            sweep).
    """
    del registry  # unused: the sidecar records the enabled rules directly
    out = Path(out_dir)
    params_path = out / "params.yaml"
    if not params_path.is_file():
        msg = f"'{out}' is not an exported sweep (no params.yaml)"
        raise ValueError(msg)
    dimensions = read_params_yaml(params_path)
    meta = _read_sweep_meta(out / SWEEP_META_FILE)
    return LoadedSweep(
        solver_name=meta["solver_name"],
        base_case=meta["base_case"],
        dimensions=dimensions,
        enabled=meta["enabled"],
        cad_model=meta["cad_model"],
    )


def _write_sweep_meta(
    path: Path,
    *,
    solver_name: str,
    base_case: str,
    enabled: list[str],
    cad_model: str,
) -> None:
    """Write the read-back sidecar (solver / base case / enabled rules / CAD model)."""
    meta = {
        "solver_name": solver_name,
        "base_case": base_case,
        "enabled": enabled,
        "cad_model": cad_model,
    }
    write_if_changed(path, json.dumps(meta, sort_keys=True, indent=2) + "\n")


def _read_sweep_meta(path: Path) -> dict[str, Any]:
    """Read the sidecar, falling back to header defaults when it is absent.

    A missing/partial sidecar (a hand-edited dir, or a sweep exported before the
    sidecar existed) still loads: the solver defaults to ``incompressibleFluid``,
    the base case to ``""``, the enabled rules to the always-present ``all`` and
    the CAD model to ``""`` — the same defaults the old Snakefile-header parser
    fell back to.
    """
    data: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict):
            data = loaded
    return {
        "solver_name": str(data.get("solver_name") or "incompressibleFluid"),
        "base_case": str(data.get("base_case", "")),
        "enabled": list(data.get("enabled") or ["all"]),
        "cad_model": str(data.get("cad_model", "")),
    }
