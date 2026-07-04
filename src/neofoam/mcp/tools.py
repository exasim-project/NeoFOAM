# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-free tool logic — plain ``f(solver, ...)`` functions returning DTOs.

These wrap the solver's case-free configuration seams
(:mod:`neofoam.framework.solver.configurations` + :mod:`neofoam.io`) and the
case-fill API into JSON-serializable Pydantic DTOs. They take ``solver`` as a
plain argument and import **no** ``fastmcp``/``fastapi`` — the protocol layer in
:mod:`neofoam.mcp.server` wraps each one in a ``@mcp.tool`` decorator. Keeping
the logic here (not in closures) makes it unit-testable without the ``mcp`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from anyio import to_thread
from pydantic import ValidationError

from neofoam.agent.case_fill import (
    build_case_agent,
    build_case_output_model,
    load_case_from_disk,
    read_case_text,
    save_case as _save_case,
)
from neofoam.framework.solver.configurations import (
    _snake_case,
    configurations,
    model_catalog as _model_catalog,
)
from neofoam.io import default_values, rjsf_uischema
from neofoam.mcp.dto import (
    CaseSpecDTO,
    CaseTextDTO,
    ConfigInfoDTO,
    ConfigSchemaDTO,
    FindingDTO,
    ModelEntryDTO,
    PatchDTO,
    SaveResultDTO,
    ToolInfoDTO,
    ValidationReportDTO,
)
from neofoam.mcp.registry import list_solver_names

INTROSPECTION_TOOL_NAMES: tuple[str, ...] = (
    "list_solvers",
    "model_catalog",
    "tool_catalog",
    "list_configs",
    "config_schema",
)
GEOMETRY_TOOL_NAMES: tuple[str, ...] = ("case_patches",)
VALIDATION_TOOL_NAMES: tuple[str, ...] = ("validate_case",)
SCAFFOLDING_TOOL_NAMES: tuple[str, ...] = ("read_case", "load_case", "save_case")
FILL_TOOL_NAMES: tuple[str, ...] = ("fill_case",)
ALL_TOOL_NAMES: tuple[str, ...] = (
    INTROSPECTION_TOOL_NAMES
    + GEOMETRY_TOOL_NAMES
    + VALIDATION_TOOL_NAMES
    + SCAFFOLDING_TOOL_NAMES
    + FILL_TOOL_NAMES
)

# OpenFOAM constraint patch types: the field's patchField type MUST equal the mesh
# patch type (unlike a regular patch/wall, which accepts any BC).
_CONSTRAINT_PATCH_TYPES = frozenset(
    {"empty", "symmetry", "symmetryPlane", "wedge", "cyclic", "cyclicAMI"}
)


# -- introspection (case-free) ------------------------------------------------


def _describe(obj: Any, *, fallback: str | None = None) -> str | None:
    """First line of ``obj``'s docstring (a one-line purpose), else ``fallback``."""
    doc = (getattr(obj, "__doc__", None) or "").strip()
    return doc.split("\n", 1)[0].strip() if doc else fallback


def list_solvers() -> list[str]:
    """Known solver spec names."""
    return list_solver_names()


def model_catalog(solver: Any) -> list[ModelEntryDTO]:
    """Every model of ``solver`` with its required flag + owned configs."""
    return [ModelEntryDTO.from_entry(e) for e in _model_catalog(solver)]


def tool_catalog(solver: Any) -> list[ToolInfoDTO]:
    """Every registered preprocessing tool, its entry schema, and the config it reads.

    The tools come from the process-wide registry; ``solver`` supplies the config
    surface so each tool's ``dict_file`` is linked to the config class that writes it
    (``blockMesh`` → ``BlockMeshDictConfig``). Lets an agent fill ``PreprocessConfig``
    (the ``tool`` names + ``depends_on`` chaining) AND know which config to author for
    each tool, instead of guessing.
    """
    import neofoam.tools  # noqa: F401  (import populates the tool registry)
    from neofoam.tools.registry import available_tools

    file_to_config = {
        cls.io_config.file: cls.__name__
        for cls in configurations(solver).classes
        if cls.io_config is not None
    }

    out: list[ToolInfoDTO] = []
    for tool in available_tools():
        step = tool.step_config_type
        dict_file = None
        if step is not None and "dict_file" in step.model_fields:
            default = step.model_fields["dict_file"].default
            dict_file = default if isinstance(default, str) else None
        out.append(
            ToolInfoDTO(
                name=tool.name,
                description=_describe(step),  # the step config's one-line docstring
                step_schema=step.model_json_schema() if step is not None else {},
                dict_file=dict_file,
                config=file_to_config.get(dict_file) if dict_file else None,
            )
        )
    return out


def list_configs(solver: Any) -> list[ConfigInfoDTO]:
    """Every config class ``solver`` may consume (name/cls_name/file)."""
    out: list[ConfigInfoDTO] = []
    for cls in configurations(solver).classes:
        io = cls.io_config
        file = io.file if io is not None else None
        out.append(
            ConfigInfoDTO(
                name=_snake_case(cls.__name__),
                cls_name=cls.__name__,
                file=file,
                # synthesised field/fv configs carry no docstring; fall back to the file
                description=_describe(
                    cls, fallback=f"Config for {file}" if file else None
                ),
            )
        )
    return out


def config_schema(solver: Any, name: str) -> ConfigSchemaDTO:
    """JSON Schema + rjsf ui-schema + defaults for one config class of ``solver``.

    Accepts either the class name (``ControlDictConfig``) or the snake-case name
    (``control_dict_config``) that :func:`list_configs` advertises, so a caller can
    round-trip a name it got from ``list_configs`` back into ``config_schema``.
    """
    cfg = configurations(solver)
    by_snake = {_snake_case(cls.__name__): cls for cls in cfg.classes}
    if name in by_snake:
        cls: Any = by_snake[name]
    else:
        try:
            cls = cfg[name]
        except KeyError as exc:
            raise ValueError(f"unknown config {name!r}; known: {cfg.names}") from exc
    json_schema = cls.model_json_schema()
    return ConfigSchemaDTO(
        name=name,
        json_schema=json_schema,
        ui_schema=rjsf_uischema(json_schema),
        defaults=default_values(cls),
    )


# -- case geometry ------------------------------------------------------------


def case_patches(case_dir: str) -> list[PatchDTO]:
    """Boundary patches (name + role) of a staged case, from ``<case>/manifest.json``.

    Lets an agent author boundary conditions without inventing patch names or roles:
    the geometry is a given, extracted upstream and written to the manifest.
    """
    from neofoam.workflow.patch_set import PatchSet

    _require_case_dir(case_dir)
    manifest = Path(case_dir) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            f"no geometry manifest at {manifest} (stage the geometry first)"
        )
    patch_set = PatchSet.load(manifest)
    return [
        PatchDTO(name=p.name, role=p.role.value, stl=p.stl) for p in patch_set.patches
    ]


# -- case validation (static pre-flight) --------------------------------------


def _mesh_patch_types(case: Path) -> dict[str, str]:
    """{patch name -> mesh patch type} from blockMeshDict boundary + snappy surfaces."""
    from neofoam.tools.block_mesh import BlockMeshDictConfig
    from neofoam.tools.snappy_hex_mesh import SnappyHexMeshDictConfig

    types: dict[str, str] = {}
    bmd = case / "system" / "blockMeshDict"
    if bmd.is_file():
        try:
            block = BlockMeshDictConfig.load(case_dir=bmd)
            if isinstance(block.boundary, list):
                for patch in block.boundary:
                    types[patch.name] = patch.type
        except Exception:
            pass
    shm = case / "system" / "snappyHexMeshDict"
    if shm.is_file():
        try:
            snappy = SnappyHexMeshDictConfig.load(case_dir=shm)
            cmc = snappy.castellatedMeshControls
            surfaces = (
                cmc.get("refinementSurfaces", {}) if isinstance(cmc, dict) else {}
            )
            for name, info in surfaces.items():
                patch_info = info.get("patchInfo") if isinstance(info, dict) else None
                if isinstance(patch_info, dict) and "type" in patch_info:
                    types[name] = str(patch_info["type"])
        except Exception:
            pass
    return types


def _dict_subentries(path: Path, section: str) -> dict[str, dict[str, str]]:
    """Read one dict section into ``{sub-name -> {leaf key -> value}}`` (best effort)."""
    import pybFoam as pyf

    out: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return out
    try:
        root = pyf.dictionary.read(str(path))
        if not root.found(section):
            return out
        block = root.subDict(section)
        for key in block.toc():
            name = str(key)
            if not block.isDict(name):
                continue
            sub = block.subDict(name)
            out[name] = {
                str(k): str(sub.get[str](str(k)))
                for k in sub.toc()
                if not sub.isDict(str(k))
            }
    except Exception:
        pass
    return out


def _scheme(path: Path, section: str, key: str) -> Optional[str]:
    """The value of one ``fvSchemes`` entry (e.g. ``divSchemes``/``div(phi,U)``)."""
    import pybFoam as pyf

    if not path.is_file():
        return None
    try:
        root = pyf.dictionary.read(str(path))
        if not root.found(section):
            return None
        block = root.subDict(section)
        for key_tok in block.toc():
            if str(key_tok) == key:
                return str(block.get[str](str(key_tok)))
    except Exception:
        pass
    return None


def _is_boussinesq(case: Path) -> bool:
    """Boussinesq buoyancy is active when transportProperties carries beta+TRef.

    Mirrors :func:`boussinesq.detect_model` so ``validate_case`` requires the
    same gravity file the buoyancy build reads at init.
    """
    import pybFoam as pyf

    path = case / "constant" / "transportProperties"
    if not path.is_file():
        return False
    try:
        props = pyf.dictionary.read(str(path))
        return bool(props.found("beta") and props.found("TRef"))
    except Exception:
        return False


def _turbulence_type(case: Path) -> Optional[str]:
    """``simulationType`` from constant/turbulenceProperties (``laminar``/``RAS``/…)."""
    import pybFoam as pyf

    path = case / "constant" / "turbulenceProperties"
    if not path.is_file():
        return None
    try:
        props = pyf.dictionary.read(str(path))
        if props.found("simulationType"):
            return str(props.get[str]("simulationType"))
    except Exception:
        pass
    return None


def validate_case(solver: Any, case_dir: str) -> ValidationReportDTO:
    """Static pre-flight of an authored case — no OpenFOAM run.

    Catches the common ways a case fails at run time: a missing config file, a
    boundary condition whose type does not match a *constraint* mesh patch (the
    empty-vs-symmetry class of error), a GAMG solver with no ``smoother`` or a missing
    ``<field>Final`` entry, an unbounded ``div(phi,U)``, a Boussinesq case missing
    ``constant/g``, and a wall-function BC in a laminar case. Returns a report whose
    ``findings`` carry a concrete fix so an agent can repair the case before meshing.
    """
    _require_case_dir(case_dir)
    case = Path(case_dir)
    findings: list[FindingDTO] = []

    def err(file: str, message: str, fix: Optional[str] = None) -> None:
        findings.append(FindingDTO(level="error", file=file, message=message, fix=fix))

    # completeness + loadability of the core config-bound files
    cfg = configurations(solver)
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
            err(rel, "required file is absent")
        else:
            try:
                loader.load(case_dir=case)
            except Exception as exc:  # present but does not validate
                err(rel, f"does not load: {exc}")
    for rel in ("system/fvSchemes", "system/fvSolution", "0/U", "0/p"):
        if not (case / rel).is_file():
            err(rel, "required file is absent")

    # boundary-condition <-> mesh-patch-type consistency (empty/symmetry/cyclic/…)
    patch_types = _mesh_patch_types(case)
    for field in ("U", "p", "T", "p_rgh", "alphat"):
        boundary = _dict_subentries(case / "0" / field, "boundaryField")
        for patch, entry in boundary.items():
            mesh_type = patch_types.get(patch)
            bc_type = entry.get("type")
            if mesh_type in _CONSTRAINT_PATCH_TYPES and bc_type != mesh_type:
                err(
                    f"0/{field}",
                    f"patch '{patch}' has BC type '{bc_type}' but the mesh patch type "
                    f"is '{mesh_type}'",
                    fix=f"set '{patch}' boundary type to {mesh_type}",
                )

    # fvSolution: GAMG needs a smoother; PIMPLE needs a <field>Final per solved field
    solvers = _dict_subentries(case / "system" / "fvSolution", "solvers")
    for name, entry in solvers.items():
        if entry.get("solver") == "GAMG" and "smoother" not in entry:
            err(
                "system/fvSolution",
                f"solver '{name}' is GAMG but has no smoother",
                fix=f"add a smoother to '{name}' (e.g. GaussSeidel)",
            )
    for name in solvers:
        if not name.endswith("Final") and f"{name}Final" not in solvers:
            err(
                "system/fvSolution",
                f"PIMPLE needs a '{name}Final' solver entry (missing)",
                fix=f"add '{name}Final' (same settings as '{name}', relTol 0)",
            )

    # Boussinesq buoyancy reads constant/g at init to build gh/ghf; without it
    # the solver aborts with "cannot find file constant/g".
    if _is_boussinesq(case) and not (case / "constant" / "g").is_file():
        err(
            "constant/g",
            "Boussinesq buoyancy is active (beta+TRef in transportProperties) but "
            "constant/g is missing",
            fix="author a GravityConfig (constant/g), e.g. value (0 -9.81 0)",
        )

    # wall-function BCs need a turbulence model; they are invalid in a laminar case
    # (e.g. compressible::alphatWallFunction on alphat with simulationType laminar).
    if _turbulence_type(case) == "laminar":
        for field in ("alphat", "nut", "k", "epsilon", "omega", "nuTilda"):
            boundary = _dict_subentries(case / "0" / field, "boundaryField")
            for patch, entry in boundary.items():
                bc_type = entry.get("type", "")
                if "WallFunction" in bc_type:
                    err(
                        f"0/{field}",
                        f"patch '{patch}' uses '{bc_type}' but the case is laminar "
                        "(wall functions require a turbulence model)",
                        fix=f"use 'calculated' (value uniform 0) for '{patch}' when laminar",
                    )

    # fvSchemes: an unbounded convection scheme is a common divergence cause
    div = _scheme(case / "system" / "fvSchemes", "divSchemes", "div(phi,U)")
    if div is not None and div.strip() in ("Gauss linear", "linear"):
        findings.append(
            FindingDTO(
                level="warning",
                file="system/fvSchemes",
                message="div(phi,U) uses an unbounded scheme (Gauss linear) — may diverge",
                fix="use a bounded scheme, e.g. Gauss linearUpwind grad(U)",
            )
        )

    ok = not any(f.level == "error" for f in findings)
    return ValidationReportDTO(ok=ok, findings=findings)


# -- case scaffolding ---------------------------------------------------------


def _require_case_dir(case_dir: str) -> None:
    """Raise a clear error when ``case_dir`` is missing or not a directory."""
    if not Path(case_dir).is_dir():
        raise ValueError(f"case_dir does not exist or is not a directory: {case_dir!r}")


def read_case(solver: Any, case_dir: str) -> CaseTextDTO:
    """Read every config-bound file of a case as raw text.

    Raises ``ValueError`` when ``case_dir`` is missing or not a directory.
    """
    _require_case_dir(case_dir)
    return CaseTextDTO(files=read_case_text(case_dir, solver=solver))


def load_case(solver: Any, case_dir: str) -> CaseSpecDTO:
    """Load each present config from disk into an aggregate CaseSpec dump.

    Raises ``ValueError`` when ``case_dir`` is missing or not a directory.
    """
    _require_case_dir(case_dir)
    spec = load_case_from_disk(case_dir, solver=solver)
    return CaseSpecDTO(values=spec.model_dump())


def save_case(solver: Any, case_spec: dict[str, Any], target_dir: str) -> SaveResultDTO:
    """Validate ``case_spec`` against the solver's aggregate model, then write.

    Validation happens before any write, so a malformed payload leaves the
    target directory untouched.
    """
    model = build_case_output_model(solver=solver)
    try:
        validated = model(**case_spec)
    except ValidationError as exc:
        raise ValueError(f"invalid case_spec for solver: {exc}") from exc
    written = _save_case(validated, target_dir)
    return SaveResultDTO(
        target_dir=str(target_dir),
        written=[str(p) for p in written],
        case_spec=validated.model_dump(),
    )


# -- LLM case-fill ------------------------------------------------------------


def _render_fill_prompt(source_dir: str, solver: Any, extra: str | None) -> str:
    """Compose the agent prompt from a source case's dictionaries (+ optional note)."""
    parts = [
        "Fill the schema from these OpenFOAM dictionaries. Use the values"
        " present; leave irrelevant configs null.\n"
    ]
    if extra:
        parts.append(f"\nAdditional instructions: {extra}\n")
    for rel, text in read_case_text(source_dir, solver=solver).items():
        if text:
            parts.append(f"\n=== {rel} ===\n{text}")
    return "".join(parts)


async def fill_case(
    solver: Any,
    source_dir: str,
    target_dir: str,
    prompt: str | None = None,
    *,
    model_name: str = "claude-haiku-4-5",
    agent_factory: Callable[..., Any] = build_case_agent,
) -> SaveResultDTO:
    """Fill a target case's configs from a source case via the LLM agent.

    ``agent_factory`` is the injection seam: it returns an object with a blocking
    ``run_sync(prompt)`` method whose ``.output`` is a validated aggregate CaseSpec.
    The default builds a pydantic-ai agent; tests pass a network-free stub.
    """
    agent = agent_factory(solver=solver, model_name=model_name)
    prompt_body = _render_fill_prompt(source_dir, solver, prompt)
    # Offload the blocking agent.run_sync off the server event loop: pydantic-ai's
    # run_sync calls asyncio.run, which raises if a loop is already running here.
    result = await to_thread.run_sync(lambda: agent.run_sync(prompt_body))
    case_spec = result.output
    written = _save_case(case_spec, target_dir)
    return SaveResultDTO(
        target_dir=str(target_dir),
        written=[str(p) for p in written],
        case_spec=case_spec.model_dump(),
    )
