# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Isolated solver-correctness checks over a staged case.

Each ``check_*`` reads one aspect of a case via :mod:`neofoam.io.dictread` (or a
config load) and yields :class:`Finding` objects. A leaf that cannot be read
(:class:`~neofoam.io.dictread.Unreadable`) becomes an *error* finding —
"couldn't-check ⇒ error" — never a downgraded-to-absent silent pass. ``configurations``
and ``pybFoam``/``neofoam.tools`` are imported lazily so ``import
neofoam.framework.validation`` stays ``pybFoam``-free.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union

from neofoam.framework.validation.registry import (
    CaseContext,
    CheckRegistry,
    Finding,
    ValidationReport,
)
from neofoam.io.dictread import (
    Leaf,
    Unreadable,
    Value,
    read_entry,
    read_keys,
    read_section,
    read_toplevel,
)

__all__ = [
    "mesh_patch_types",
    "is_boussinesq",
    "turbulence_type",
    "check_required_files",
    "check_constraint_patches",
    "check_gamg_smoother",
    "check_pimple_final",
    "check_boussinesq_gravity",
    "check_laminar_wall_functions",
    "check_div_scheme",
    "default_registry",
    "validate",
]

# OpenFOAM constraint patch types: the field's patchField type MUST equal the mesh
# patch type (unlike a regular patch/wall, which accepts any BC).
_CONSTRAINT_PATCH_TYPES = frozenset(
    {"empty", "symmetry", "symmetryPlane", "wedge", "cyclic", "cyclicAMI"}
)


def _text(leaf: Optional[Leaf]) -> Optional[str]:
    """Rendered text of a readable leaf; ``None`` for a genuinely-absent leaf.

    Refuses an :class:`Unreadable`: callers escalate it via :func:`_leaf_or_error`
    first, so a forgotten guard *raises* (→ :meth:`CheckRegistry.run` turns it into an
    error finding) rather than silently downgrading unreadable to absent — closing the
    footgun structurally, not by per-check discipline.
    """
    if isinstance(leaf, Unreadable):
        raise TypeError("Unreadable leaf must be escalated to a Finding before _text")
    return leaf.text if isinstance(leaf, Value) else None


def _error(file: str, message: str, fix: Optional[str] = None) -> Finding:
    return Finding(level="error", file=file, message=message, fix=fix)


def _leaf_or_error(
    leaf: Optional[Leaf], file: str, message: str, *, fix: str
) -> tuple[Optional[str], Optional[Finding]]:
    """Text of a readable/absent leaf, or an escalation :class:`Finding` for an
    ``Unreadable`` one.

    The single home for "couldn't-read a leaf ⇒ error", so no check re-implements (or
    forgets) the escalation.
    """
    if isinstance(leaf, Unreadable):
        return None, _error(file, f"{message} ({leaf.reason})", fix=fix)
    return _text(leaf), None


def _boussinesq_or_error(
    case: Path,
) -> tuple[Optional[bool], Optional[Finding]]:
    """Boussinesq state, or an escalation Finding when it cannot be determined.

    The helper analogue of :func:`_leaf_or_error` for :func:`is_boussinesq`, so the
    two Boussinesq-dependent checks share one escalation site.
    """
    buoyant = is_boussinesq(case)
    if isinstance(buoyant, Unreadable):
        return None, _error(
            "constant/transportProperties",
            f"could not determine Boussinesq state ({buoyant.reason})",
            fix="ensure constant/transportProperties parses",
        )
    return buoyant, None


# -- non-leaf reads (moved verbatim from mcp/tools.py) -------------------------


# Geometry-manifest patch roles that map to an OpenFOAM constraint patch type, so a
# staged (but not-yet-meshed) case still exposes its constraint patches to the check.
_ROLE_TO_CONSTRAINT_TYPE = {"symmetry": "symmetry", "empty": "empty"}


def _manifest_constraint_types(case: Path) -> dict[str, str]:
    """Constraint patch types the geometry ``manifest.json`` declares, or ``{}``.

    Lets :func:`check_constraint_patches` work *before* the mesh is built (F4): the
    manifest (written by ``import_geometry``) records each patch's CFD role, and the
    constraint roles (``symmetry``/``empty``) map to the same mesh patch type. A missing
    or unreadable manifest yields ``{}`` — it is an optional pre-mesh supplement, never a
    hard dependency, and the mesh dicts take precedence when both are present.
    """
    manifest = case / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        from neofoam.tooling.workflow.geometry import PatchSet

        patch_set = PatchSet.load(manifest)
    except Exception:
        return {}
    return {
        p.name: _ROLE_TO_CONSTRAINT_TYPE[p.role.value]
        for p in patch_set.patches
        if p.role.value in _ROLE_TO_CONSTRAINT_TYPE
    }


def mesh_patch_types(case: Path) -> Union[dict[str, str], Unreadable]:
    """{patch name -> mesh patch type} from blockMeshDict boundary + snappy surfaces.

    A present-but-unparsable blockMeshDict/snappyHexMeshDict escalates to an
    :class:`Unreadable` (couldn't-check ⇒ error at the call site) instead of silently
    downgrading to an empty set — the non-leaf level of the honest-validation gap.
    Routing this through :mod:`neofoam.io.dictread` is not feasible: the patch contract
    is a *config* load (a list-valued ``boundary`` + nested ``refinementSurfaces``), not
    a flat leaf section, so the read seam kept here is ``Config.load`` and its failure is
    surfaced as ``Unreadable``.

    The geometry ``manifest.json`` (when present) supplies constraint patch types too,
    so the check runs *pre-mesh* on a staged case (F4); the mesh dicts win on conflict.
    """
    from neofoam.tools.block_mesh import BlockMeshDictConfig
    from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig

    types: dict[str, str] = _manifest_constraint_types(case)
    bmd = case / "system" / "blockMeshDict"
    if bmd.is_file():
        try:
            block = BlockMeshDictConfig.load(case_dir=bmd)
        except Exception as exc:
            return Unreadable(
                reason=f"system/blockMeshDict: {str(exc).strip() or type(exc).__name__}"
            )
        if isinstance(block.boundary, list):
            for patch in block.boundary:
                types[patch.name] = patch.type
    shm = case / "system" / "snappyHexMeshDict"
    if shm.is_file():
        try:
            snappy = SnappyHexMeshDictConfig.load(case_dir=shm)
        except Exception as exc:
            return Unreadable(
                reason=(
                    "system/snappyHexMeshDict: "
                    f"{str(exc).strip() or type(exc).__name__}"
                )
            )
        cmc = snappy.castellatedMeshControls
        surfaces = cmc.get("refinementSurfaces", {}) if isinstance(cmc, dict) else {}
        for name, info in surfaces.items():
            patch_info = info.get("patchInfo") if isinstance(info, dict) else None
            if isinstance(patch_info, dict) and "type" in patch_info:
                types[name] = str(patch_info["type"])
    return types


def is_boussinesq(case: Path) -> Union[bool, Unreadable]:
    """Boussinesq buoyancy is active when transportProperties carries beta+TRef.

    Presence-read via :func:`neofoam.io.dictread.read_keys`, so a corrupt
    transportProperties escalates to :class:`Unreadable` instead of a swallowed
    ``False``. Only key presence is read (never a dimensioned value's text).
    """
    keys = read_keys(case / "constant" / "transportProperties")
    if isinstance(keys, Unreadable):
        return keys
    if keys is None:
        return False
    return "beta" in keys and "TRef" in keys


def turbulence_type(case: Path) -> Union[str, None, Unreadable]:
    """``simulationType`` from constant/turbulenceProperties (``laminar``/``RAS``/…).

    ``None`` when absent; :class:`Unreadable` when the file will not parse — via
    :func:`neofoam.io.dictread.read_toplevel`, so a corrupt turbulenceProperties
    escalates instead of silently reading as non-laminar.
    """
    leaf = read_toplevel(case / "constant" / "turbulenceProperties", "simulationType")
    if isinstance(leaf, Unreadable):
        return leaf
    if leaf is None:
        return None
    return leaf.text


# -- checks -------------------------------------------------------------------


def check_required_files(ctx: CaseContext) -> list[Finding]:
    """The core config-bound files must be present and (where present) loadable.

    The required ``0/<field>`` set is derived from the solver's *required* model
    families (via :func:`model_catalog`) rather than hardcoded — so a VoF case is
    checked for ``0/U`` / ``0/p_rgh`` / ``0/alpha.water`` (not the fluid solver's
    ``0/p``), and each solver is judged against the fields it actually owns.
    """
    from neofoam.framework.solver.configurations import configurations, model_catalog

    case = ctx.case
    findings: list[Finding] = []
    cfg = configurations(ctx.solver)
    core = {
        cfg[name].io_config.file: cfg[name]  # type: ignore[union-attr]
        for name in (
            "ControlDictConfig",
            "TransportPropertiesConfig",
            "TurbulencePropertiesConfig",
        )
        if name in cfg.names and cfg[name].io_config is not None
    }
    for rel, loader in core.items():
        if not (case / rel).is_file():
            findings.append(_error(rel, "required file is absent"))
        else:
            try:
                loader.load(case_dir=case)
            except Exception as exc:  # present but does not validate
                findings.append(_error(rel, f"does not load: {exc}"))

    # The 0/<field> files the *required* models own (dedup, declaration order).
    required_fields: list[str] = []
    for entry in model_catalog(ctx.solver):
        if not entry.required:
            continue
        for field_cls in entry.fields:
            io = getattr(field_cls, "io_config", None)
            if io is not None and io.file not in required_fields:
                required_fields.append(io.file)

    for rel in ("system/fvSchemes", "system/fvSolution", *required_fields):
        if not (case / rel).is_file():
            findings.append(_error(rel, "required file is absent"))
    return findings


def check_constraint_patches(ctx: CaseContext) -> list[Finding]:
    """A constraint mesh patch (empty/symmetry/…) requires a matching BC type."""
    findings: list[Finding] = []
    patch_types = mesh_patch_types(ctx.case)
    if isinstance(patch_types, Unreadable):
        return [
            _error(
                "system/blockMeshDict",
                f"mesh patch types could not be read ({patch_types.reason})",
                fix="ensure system/blockMeshDict (and snappyHexMeshDict) parse — a "
                "corrupt mesh dict cannot be checked",
            )
        ]
    for field_name in ("U", "p", "T", "p_rgh", "alphat"):
        boundary = read_section(ctx.case / "0" / field_name, "boundaryField")
        for patch, entry in boundary.items():
            bc_type, err = _leaf_or_error(
                entry.get("type"),
                f"0/{field_name}",
                f"patch '{patch}' boundary type could not be read",
                fix="ensure the boundaryField entry is a well-formed patch type",
            )
            if err is not None:
                findings.append(err)
                continue
            mesh_type = patch_types.get(patch)
            if mesh_type in _CONSTRAINT_PATCH_TYPES and bc_type != mesh_type:
                findings.append(
                    _error(
                        f"0/{field_name}",
                        f"patch '{patch}' has BC type '{bc_type}' but the mesh patch "
                        f"type is '{mesh_type}'",
                        fix=f"set '{patch}' boundary type to {mesh_type}",
                    )
                )
    return findings


def check_gamg_smoother(ctx: CaseContext) -> list[Finding]:
    """A GAMG solver needs a smoother; an unreadable solver type is an error."""
    findings: list[Finding] = []
    solvers = read_section(ctx.case / "system" / "fvSolution", "solvers")
    for name, entry in solvers.items():
        solver_text, err = _leaf_or_error(
            entry.get("solver"),
            "system/fvSolution",
            f"solver '{name}' type could not be read",
            fix="ensure the 'solver' entry names a solver (e.g. GAMG, PCG)",
        )
        if err is not None:
            findings.append(err)
            continue
        if solver_text == "GAMG" and "smoother" not in entry:
            findings.append(
                _error(
                    "system/fvSolution",
                    f"solver '{name}' is GAMG but has no smoother",
                    fix=f"add a smoother to '{name}' (e.g. GaussSeidel)",
                )
            )
    return findings


def _expand_group(name: str) -> list[str]:
    """Expand an OpenFOAM grouped solver key into individual field names.

    ``"(U|k|epsilon)"`` -> ``["U", "k", "epsilon"]``; a plain name returns itself.
    Surrounding quotes (regex/wordRe keys) are stripped first. This is what lets
    ``check_pimple_final`` accept grouped solver keys instead of demanding a literal
    ``"(U|k|epsilon)Final"`` entry.
    """
    stripped = name.strip().strip('"')
    m = re.fullmatch(r"\(([^)]*)\)", stripped)
    if m is not None:
        return [p for p in m.group(1).split("|") if p]
    return [stripped] if stripped else []


def check_pimple_final(ctx: CaseContext) -> list[Finding]:
    """Every solved field needs a ``<field>Final`` entry — grouped keys expanded.

    Base and Final solver keys are expanded to individual fields, so a grouped base
    ``"(U|k|epsilon)"`` is satisfied by per-field ``UFinal``/``kFinal``/``epsilonFinal``
    (or a grouped ``"(U|k|epsilon)Final"``) — no literal-string false-fail. In a
    Boussinesq case the vestigial ``p`` solver (never solved; p_rgh is) needs no Final.
    """
    findings: list[Finding] = []
    solvers = read_section(ctx.case / "system" / "fvSolution", "solvers")
    buoyant, err = _boussinesq_or_error(ctx.case)
    if err is not None:
        return [err]
    base_fields: set[str] = set()
    final_fields: set[str] = set()
    for name in solvers:
        stripped = name.strip().strip('"')
        if stripped.endswith("Final"):
            final_fields.update(_expand_group(stripped[: -len("Final")]))
        else:
            base_fields.update(_expand_group(stripped))
    for field_name in sorted(base_fields):
        if buoyant and field_name == "p":
            continue
        if field_name not in final_fields:
            findings.append(
                _error(
                    "system/fvSolution",
                    f"PIMPLE needs a '{field_name}Final' solver entry (missing)",
                    fix=f"add '{field_name}Final' (same settings as '{field_name}', "
                    "relTol 0)",
                )
            )
    return findings


def check_boussinesq_gravity(ctx: CaseContext) -> list[Finding]:
    """A Boussinesq case reads constant/g at init; without it the solver aborts."""
    buoyant, err = _boussinesq_or_error(ctx.case)
    if err is not None:
        return [err]
    if buoyant and not (ctx.case / "constant" / "g").is_file():
        return [
            _error(
                "constant/g",
                "Boussinesq buoyancy is active (beta+TRef in transportProperties) but "
                "constant/g is missing",
                fix="author a GravityConfig (constant/g), e.g. value (0 -9.81 0)",
            )
        ]
    return []


def check_laminar_wall_functions(ctx: CaseContext) -> list[Finding]:
    """Wall-function BCs are invalid in a laminar case; an unreadable type is an error."""
    findings: list[Finding] = []
    turb = turbulence_type(ctx.case)
    if isinstance(turb, Unreadable):
        return [
            _error(
                "constant/turbulenceProperties",
                f"turbulence type could not be read ({turb.reason})",
                fix="ensure constant/turbulenceProperties parses",
            )
        ]
    if turb != "laminar":
        return findings
    for field_name in ("alphat", "nut", "k", "epsilon", "omega", "nuTilda"):
        boundary = read_section(ctx.case / "0" / field_name, "boundaryField")
        for patch, entry in boundary.items():
            bc_text, err = _leaf_or_error(
                entry.get("type"),
                f"0/{field_name}",
                f"patch '{patch}' boundary type could not be read",
                fix="ensure the boundaryField entry is a well-formed patch type",
            )
            if err is not None:
                findings.append(err)
                continue
            bc_type = bc_text or ""
            if "WallFunction" in bc_type:
                findings.append(
                    _error(
                        f"0/{field_name}",
                        f"patch '{patch}' uses '{bc_type}' but the case is laminar "
                        "(wall functions require a turbulence model)",
                        fix=f"use 'calculated' (value uniform 0) for '{patch}' when "
                        "laminar",
                    )
                )
    return findings


def check_div_scheme(ctx: CaseContext) -> list[Finding]:
    """An unbounded div(phi,U) is a warning; a present-but-unreadable scheme is error."""
    div, err = _leaf_or_error(
        read_entry(ctx.case / "system" / "fvSchemes", "divSchemes", "div(phi,U)"),
        "system/fvSchemes",
        "div(phi,U) scheme could not be read",
        fix="ensure div(phi,U) names a scheme, e.g. Gauss linearUpwind grad(U)",
    )
    if err is not None:
        return [err]
    if div is not None and div.strip() in ("Gauss linear", "linear"):
        return [
            Finding(
                level="warning",
                file="system/fvSchemes",
                message="div(phi,U) uses an unbounded scheme (Gauss linear) — may diverge",
                fix="use a bounded scheme, e.g. Gauss linearUpwind grad(U)",
            )
        ]
    return []


def default_registry() -> CheckRegistry:
    """The standard incompressible-case check set, in report order."""
    reg = CheckRegistry()
    reg.add("required-files", check_required_files)
    reg.add("constraint-patches", check_constraint_patches)
    reg.add("gamg-smoother", check_gamg_smoother)
    reg.add("pimple-final", check_pimple_final)
    reg.add("boussinesq-gravity", check_boussinesq_gravity)
    reg.add("laminar-wall-functions", check_laminar_wall_functions)
    reg.add("div-scheme", check_div_scheme)
    return reg


def validate(solver: Any, case_dir: Union[Path, str]) -> ValidationReport:
    """Run the default check registry over ``case_dir`` → a :class:`ValidationReport`."""
    ctx = CaseContext(case=Path(case_dir), solver=solver)
    return default_registry().run(ctx)
