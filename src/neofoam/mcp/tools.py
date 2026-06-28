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
from typing import Any, Callable

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
    toggle_models as _toggle_models,
)
from neofoam.io import default_values, rjsf_uischema
from neofoam.mcp.dto import (
    CaseSpecDTO,
    CaseTextDTO,
    ConfigInfoDTO,
    ConfigSchemaDTO,
    ModelEntryDTO,
    SaveResultDTO,
    ToggleModelDTO,
)
from neofoam.mcp.registry import list_solver_names

INTROSPECTION_TOOL_NAMES: tuple[str, ...] = (
    "list_solvers",
    "model_catalog",
    "toggle_models",
    "list_configs",
    "config_schema",
)
SCAFFOLDING_TOOL_NAMES: tuple[str, ...] = ("read_case", "load_case", "save_case")
FILL_TOOL_NAMES: tuple[str, ...] = ("fill_case",)
ALL_TOOL_NAMES: tuple[str, ...] = (
    INTROSPECTION_TOOL_NAMES + SCAFFOLDING_TOOL_NAMES + FILL_TOOL_NAMES
)


# -- introspection (case-free) ------------------------------------------------


def list_solvers() -> list[str]:
    """Known solver spec names."""
    return list_solver_names()


def model_catalog(solver: Any) -> list[ModelEntryDTO]:
    """Every model of ``solver`` with its required flag + owned configs."""
    return [ModelEntryDTO.from_entry(e) for e in _model_catalog(solver)]


def toggle_models(solver: Any) -> list[ToggleModelDTO]:
    """Optional models of ``solver`` flagged as on/off toggles."""
    return [ToggleModelDTO.from_toggle(t) for t in _toggle_models(solver)]


def list_configs(solver: Any) -> list[ConfigInfoDTO]:
    """Every config class ``solver`` may consume (name/cls_name/file)."""
    out: list[ConfigInfoDTO] = []
    for cls in configurations(solver).classes:
        io = cls.io_config
        out.append(
            ConfigInfoDTO(
                name=_snake_case(cls.__name__),
                cls_name=cls.__name__,
                file=io.file if io is not None else None,
            )
        )
    return out


def config_schema(solver: Any, name: str) -> ConfigSchemaDTO:
    """JSON Schema + rjsf ui-schema + defaults for one config class of ``solver``."""
    cfg = configurations(solver)
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
