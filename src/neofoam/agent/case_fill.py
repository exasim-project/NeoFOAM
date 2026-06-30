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
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from neofoam.framework.solver.configurations import _snake_case, configurations
from neofoam.io import BaseConfig

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
    """Lazy import so the agent module stays importable without OpenFOAM."""
    from neofoam.solver.incompressibleFluid.incompressibleFluid import (
        incompressibleFluid,
    )

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
) -> BaseModel:
    """Build a ``CaseSpec`` by loading each config from disk (no LLM).

    Skips classes without an ``@IOStrategy`` binding and classes whose file
    is not present — the resulting aggregate carries ``None`` in those
    fields, which :func:`save_case` then skips. This is the deterministic
    roundtrip path used by tests.
    """
    solver = solver or _solver()
    output_model = output_model or build_case_output_model(solver=solver)
    case_dir = Path(case_dir)

    values: dict[str, Any] = {}
    for cls in configurations(solver):
        io = getattr(cls, "io_config", None)
        if io is None:
            continue
        if not (case_dir / io.file).exists():
            continue
        try:
            values[_snake_case(cls.__name__)] = cls.load(case_dir=case_dir)
        except Exception:
            # File present but doesn't validate against this schema (e.g. a
            # turbulenceProperties that selects a model we don't represent).
            # Leave as None; the caller can surface the gap.
            continue
    return output_model(**values)


def save_case(
    case_spec: BaseModel,
    target_case: Union[Path, str],
) -> list[Path]:
    """Write every populated config in ``case_spec`` to ``target_case``.

    Delegates to :func:`neofoam.io.write_configs`, which groups configs that
    target the same file (e.g. ``TransportProperties`` + ``Boussinesq``) and
    writes each file once, so co-owners don't clobber each other.
    """
    from neofoam.io import write_configs

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
    **agent_kwargs: Any,
) -> Any:
    """Build a pydantic-ai ``Agent`` returning a filled aggregate ``CaseSpec``.

    By default uses the Anthropic backend with ``claude-haiku-4-5``; pass a
    pre-built ``model`` to use the local-Ollama scaffold from
    :mod:`neofoam.agent.agent` instead. Extra ``agent_kwargs`` are forwarded
    to :class:`pydantic_ai.Agent`.
    """
    from pydantic_ai import Agent

    if model is None:
        from pydantic_ai.models.anthropic import AnthropicModel

        model = AnthropicModel(model_name)

    output_type = build_case_output_model(solver=solver, model_name=output_model_name)
    return Agent(
        model,
        output_type=output_type,
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
    """
    solver = solver or _solver()
    source_case = Path(source_case)
    target_case = Path(target_case)

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
