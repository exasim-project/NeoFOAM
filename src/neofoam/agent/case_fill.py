# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Fill an ``incompressibleFluid`` case's config files with an LLM agent.

Layered API so the same machinery serves the CLI, an LLM workflow, and a
deterministic test path:

- :func:`build_case_output_model` — aggregate ``CaseSpec`` pydantic model,
  one ``Optional`` field per ``BaseConfig`` the solver may consume. Suitable
  as a pydantic-ai ``output_type``.
- :func:`load_case_from_disk` — read each config from a source case via the
  registered IO strategy (no LLM, deterministic, used by tests).
- :func:`save_case` — write every non-None field of a ``CaseSpec`` instance
  to the target case via ``save_configs``.
- :func:`build_case_agent` — pydantic-ai ``Agent`` whose ``output_type`` is
  the aggregate; backed by Anthropic by default, falls back to the local
  Ollama scaffold via ``base_url``.
- :func:`fill_case` — one-shot: copy the on-disk mesh + ``0/``/``0.orig``
  fields from ``source_case`` to ``target_case``, then either disk-roundtrip
  the configs (no agent) or have an agent rewrite them.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, Union

import pybFoam as pyf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from neofoam.framework.solver.configurations import _snake_case, configurations
from neofoam.io import BaseConfig, write_configs
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.anthropic import AnthropicModel
except ModuleNotFoundError:  # optional [agent] extra
    Agent = None  # type: ignore[assignment,misc]
    AnthropicModel = None  # type: ignore[assignment,misc]

#: Actionable message when the optional ``[agent]`` extra is missing.
_MISSING_PYDANTIC_AI = (
    "the agent feature needs pydantic-ai; install with: pip install neofoam[agent]"
)

#: A key's address in a dict file, from the file's root.
_KeyPath = tuple[str, ...]

__all__ = [
    "DEFAULT_CASE_SYSTEM_PROMPT",
    "build_case_agent",
    "build_case_output_model",
    "case_spec_to_configs",
    "fill_case",
    "load_case_from_disk",
    "read_case_text",
    "save_case",
]


DEFAULT_CASE_SYSTEM_PROMPT = (
    "You scaffold an OpenFOAM case for the ``incompressibleFluid`` solver."
    " Fill ONLY the configs needed for the user's case description; leave"
    " every other field as null. Always include ControlDictConfig,"
    " TransportPropertiesConfig and TurbulencePropertiesConfig. Include"
    " Pimple_fvSchemes / Pimple_fvSolution whenever PIMPLE pressure-velocity"
    " coupling is used (which is always, for this solver). Set Boussinesq"
    " configs only when the user explicitly asks for buoyancy."
)


def _solver() -> Any:
    """Return the incompressibleFluid solver spec."""
    return incompressibleFluid


def build_case_output_model(
    *,
    solver: Optional[Any] = None,
    model_name: str = "CaseSpec",
) -> type[BaseModel]:
    """Return an aggregate pydantic model with one ``Optional`` field per config.

    Every field defaults to ``None`` so a single agent prompt can fill only
    the subset the case needs (e.g. no ``Boussinesq*`` for an isothermal
    flow). Field names are the snake-case class names, matching the
    convention of :meth:`Configurations.as_output_model`.
    """
    solver = solver or _solver()
    classes = list(configurations(solver))
    fields: dict[str, Any] = {
        _snake_case(cls.__name__): (
            Optional[cls],
            Field(default=None, description=cls.__doc__ or cls.__name__),
        )
        for cls in classes
    }
    return create_model(
        model_name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )


def case_spec_to_configs(case_spec: BaseModel) -> list[BaseConfig]:
    """Pull every non-``None`` ``BaseConfig`` instance out of an aggregate.

    The aggregate is whatever :func:`build_case_output_model` produced; this
    walks its fields in declaration order and filters to BaseConfig values.
    """
    out: list[BaseConfig] = []
    for name in type(case_spec).model_fields:
        value = getattr(case_spec, name)
        if isinstance(value, BaseConfig):
            out.append(value)
    return out


def read_case_text(
    case_dir: Union[Path, str],
    *,
    solver: Optional[Any] = None,
) -> dict[str, str]:
    """Read every config-bound file in ``case_dir`` as raw text.

    Keys are the registered IO file paths (e.g. ``"system/controlDict"``);
    values are the file contents (empty string when the file is missing).
    This is what gets fed to the LLM so it can extract structured values.
    """
    solver = solver or _solver()
    case_dir = Path(case_dir)
    seen: dict[str, str] = {}
    for cls in configurations(solver):
        io = getattr(cls, "io_config", None)
        if io is None:
            continue
        rel = io.file
        if rel in seen:
            continue
        path = case_dir / rel
        seen[rel] = path.read_text() if path.exists() else ""
    return seen


def load_case_from_disk(
    case_dir: Union[Path, str],
    *,
    solver: Optional[Any] = None,
    output_model: Optional[type[BaseModel]] = None,
    warnings: Optional[list[dict[str, str]]] = None,
) -> BaseModel:
    """Build a ``CaseSpec`` by loading each config from disk (no LLM).

    Skips classes without an ``@IOStrategy`` binding and classes whose file
    is not present — the resulting aggregate carries ``None`` in those
    fields, which :func:`save_case` then skips. This is the deterministic
    roundtrip path used by tests.

    Pass a ``warnings`` list to surface configs that are *present on disk but
    don't validate* (e.g. a ``turbulenceProperties`` selecting a model we don't
    represent), or that lack some of the keys only they require: each is appended
    as ``{"file", "config", "reason"}`` instead of only silently nulling the
    field — so a caller can report the gap (F5).
    """
    solver = solver or _solver()
    output_model = output_model or build_case_output_model(solver=solver)
    case_dir = Path(case_dir)

    classes = list(configurations(solver))
    values: dict[str, Any] = {}
    for cls in classes:
        io = getattr(cls, "io_config", None)
        if io is None:
            continue
        if not (case_dir / io.file).exists():
            continue
        try:
            values[_snake_case(cls.__name__)] = cls.load(case_dir=case_dir)
        except Exception as exc:
            # File present but doesn't validate against this schema. Several configs
            # slice one file (a plain case's shared ``controlDict`` has no ``maxCo``
            # for ``CourantConfig``, a non-buoyant ``fvSchemes`` no ``div(phi,T)`` for
            # the boussinesq slice), so a slice the file carries nothing of is absent:
            # leave the field ``None`` silently. Anything else is surfaced (F5).
            reason = _invalid_reason(exc, _own_keys(cls, classes), case_dir / io.file)
            if warnings is not None and reason:
                warnings.append({"file": io.file, "config": cls.__name__, "reason": reason})
            continue
    return output_model(**values)


def _invalid_reason(exc: Exception, own_keys: set[_KeyPath], path: Path) -> Optional[str]:
    """Why a config that failed to load is present-but-invalid; ``None`` when it is absent."""
    if not _is_incomplete(exc):
        return str(exc).strip() or type(exc).__name__
    if not isinstance(exc, ValidationError) or not _holds_any(path, own_keys):
        return None
    return "missing " + ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())


def _is_incomplete(exc: Exception) -> bool:
    """True when ``exc`` only says that keys are *missing*, not that a value is invalid.

    That is a pydantic ``ValidationError`` whose every error is a *missing required
    field*, or a ``KeyError`` from a config bound to a sub-dict whose block isn't in
    the file (e.g. ``TelemetryDictConfig`` → ``controlDict{telemetry}`` on a case with
    no telemetry block).
    """
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        return bool(errors) and all(e.get("type") == "missing" for e in errors)
    if isinstance(exc, KeyError):
        message = str(exc)
        return "Subdict" in message and "not found" in message
    return False


def _required_keys(model: type[BaseModel], prefix: _KeyPath) -> Iterator[_KeyPath]:
    """Paths of the required keys ``model`` declares; ``default`` is a shorthand, not a key."""
    for name, info in model.model_fields.items():
        path = (*prefix, info.alias or name)
        nested = info.annotation
        if isinstance(nested, type) and issubclass(nested, BaseModel):
            yield from _required_keys(nested, path)
        elif info.is_required() and path[-1] != "default":
            yield path


def _file_keys(cls: type[BaseConfig]) -> set[_KeyPath]:
    """The required keys of ``cls`` as paths from the root of its file."""
    subdict = cls.io_config.subdict if cls.io_config else None
    return set(_required_keys(cls, tuple(subdict.split(".")) if subdict else ()))


def _own_keys(cls: type[BaseConfig], classes: list[type[BaseConfig]]) -> set[_KeyPath]:
    """The required keys only ``cls`` declares for its file.

    A key another config of the same file requires as well (``solvers.p`` of the
    Pimple and the Simple slice) is no evidence of either.
    """
    rivals = [c for c in classes if c is not cls and _file(c) == _file(cls)]
    return _file_keys(cls) - {key for rival in rivals for key in _file_keys(rival)}


def _file(cls: type[BaseModel]) -> Optional[str]:
    """The file ``cls`` is bound to, if any."""
    io = getattr(cls, "io_config", None)
    return io.file if io else None


def _holds_any(path: Path, keys: set[_KeyPath]) -> bool:
    """True when the OpenFOAM dict at ``path`` spells one of ``keys`` out."""
    if not keys:
        return False
    root = pyf.dictionary.read(str(path))
    return any(_holds(root, key) for key in keys)


def _holds(node: Any, key: _KeyPath) -> bool:
    """True when the dict spells ``key`` out; ``found`` lets ``"p.*"`` answer ``p_rgh``."""
    for part in key[:-1]:
        if part not in map(str, node.toc()) or not node.isDict(part):
            return False
        node = node.subDict(part)
    return key[-1] in map(str, node.toc())


def save_case(
    case_spec: BaseModel,
    target_case: Union[Path, str],
) -> list[Path]:
    """Write every populated config in ``case_spec`` to ``target_case``.

    Delegates to :func:`neofoam.io.write_configs`, which groups configs that
    target the same file (e.g. ``TransportProperties`` + ``Boussinesq``) and
    writes each file once, so co-owners don't clobber each other.
    """
    report = write_configs(case_spec_to_configs(case_spec), target_case)
    return [Path(target_case) / file for file in report]


def _copy_static_assets(source_case: Path, target_case: Path) -> None:
    """Mirror the parts of a case the agent can't (yet) generate.

    What we copy:

    - ``constant/polyMesh`` (or ``system/blockMeshDict`` if the mesh
      hasn't been built yet) and ``system/decomposeParDict``;
    - ``0/`` and ``0.orig/`` field dictionaries — boundary conditions are
      not part of the solver's config schema today;
    - ``system/fvSchemes`` / ``system/fvSolution`` — the per-spec config
      classes only model the entries the in-tree operations declared via
      ``@<Subclass>.add(...)``; writing from scratch would drop the rest.
      Mirroring then letting ``save_case`` patch in place keeps real
      cases (which have many more entries) working. Other configs
      (``controlDict``, ``transportProperties``, ``turbulenceProperties``)
      write cleanly from the schema alone now and don't need mirroring.

    Existing target files are removed before the copy.
    """
    target_case.mkdir(parents=True, exist_ok=True)
    for rel in (
        "constant/polyMesh",
        "system/blockMeshDict",
        "system/decomposeParDict",
        "system/fvSchemes",
        "system/fvSolution",
        "0",
        "0.orig",
    ):
        src = source_case / rel
        if not src.exists():
            continue
        dst = target_case / rel
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def build_case_agent(
    *,
    solver: Optional[Any] = None,
    model: Any = None,
    model_name: str = "claude-haiku-4-5",
    system_prompt: str = DEFAULT_CASE_SYSTEM_PROMPT,
    output_model_name: str = "CaseSpec",
    output_type: Optional[type[BaseModel]] = None,
    **agent_kwargs: Any,
) -> Any:
    """Build a pydantic-ai ``Agent`` returning a filled aggregate ``CaseSpec``.

    By default uses the Anthropic backend with ``claude-haiku-4-5``; pass a
    pre-built ``model`` to use the local-Ollama scaffold from
    :mod:`neofoam.agent.agent` instead. Pass ``output_type`` to make the agent
    emit that model directly (e.g. a custom aggregate) instead of the synthesised
    per-solver ``CaseSpec``. Extra ``agent_kwargs`` are forwarded to
    :class:`pydantic_ai.Agent`.
    """
    if Agent is None:
        raise ImportError(_MISSING_PYDANTIC_AI)

    if model is None:
        if AnthropicModel is None:
            raise ImportError(_MISSING_PYDANTIC_AI)
        model = AnthropicModel(model_name)

    out_type = output_type or build_case_output_model(solver=solver, model_name=output_model_name)
    return Agent(
        model,
        output_type=out_type,
        system_prompt=system_prompt,
        **agent_kwargs,
    )


def _agent_prompt_from_source(source_case: Path, solver: Any) -> str:
    """Render the per-file source text into one prompt body."""
    files = read_case_text(source_case, solver=solver)
    parts = [
        "Fill the schema from these OpenFOAM dictionaries. Use the values"
        " present; leave irrelevant configs null.\n"
    ]
    for rel, text in files.items():
        if not text:
            continue
        parts.append(f"\n=== {rel} ===\n{text}")
    return "".join(parts)


def fill_case(
    source_case: Union[Path, str],
    target_case: Union[Path, str],
    *,
    agent: Any = None,
    solver: Optional[Any] = None,
    copy_static: bool = True,
) -> BaseModel:
    """Fill the configs of ``target_case`` from ``source_case``.

    The mesh, BC fields (``0/`` or ``0.orig/``) and decomposeParDict are
    mirrored from ``source_case`` (set ``copy_static=False`` to skip).
    Then:

    - if ``agent`` is provided, prompt it with the source-case text and use
      its validated ``CaseSpec`` to write the configs;
    - otherwise, load the configs from disk via :func:`load_case_from_disk`
      and write them out — a deterministic, LLM-free roundtrip used by
      tests and by the CLI's ``--no-llm`` mode.

    Returns the validated aggregate that was saved.

    A missing/invalid ``source_case`` raises :class:`ValueError` **before** any
    static-asset copy or agent call, so a bad source never burns an LLM API call on
    a contentless prompt (the guard the removed MCP ``fill_case`` tool used to carry).
    """
    solver = solver or _solver()
    source_case = Path(source_case)
    target_case = Path(target_case)

    if not source_case.is_dir():
        raise ValueError(f"source_case does not exist or is not a directory: {str(source_case)!r}")

    if copy_static:
        _copy_static_assets(source_case, target_case)

    if agent is None:
        case_spec = load_case_from_disk(source_case, solver=solver)
    else:
        prompt = _agent_prompt_from_source(source_case, solver)
        result = agent.run_sync(prompt)
        case_spec = result.output

    save_case(case_spec, target_case)
    return case_spec
