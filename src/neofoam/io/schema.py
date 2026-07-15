# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Config-introspection surface — schema/ui/defaults shaping + tool/model catalogs.

Pure composition over :func:`neofoam.framework.solver.configurations.configurations`:
given a resolved ``solver`` spec it produces the JSON-serializable views a frontend
(wizard, MCP host, CLI) renders — ``model_json_schema()`` + ``rjsf_uischema()`` +
``default_values()`` per config, plus the preprocessing-tool and model catalogs.
Frontend-agnostic: imports no ``mcp``/``fastmcp``. The framework/``tools`` imports are
lazy inside the functions so ``import neofoam.io`` pulls no ``pybFoam``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from neofoam.io.pydantic_schema import default_values, rjsf_uischema

__all__ = [
    "ConfigInfo",
    "ConfigSchema",
    "ToolInfo",
    "ModelSummary",
    "list_configs",
    "config_schema",
    "tool_catalog",
    "model_catalog",
]


class ConfigInfo(BaseModel):
    """One config class a solver may consume (snake name + class name + file + origin).

    ``origin`` tells a caller *whether it must author this config*:

    * ``"solver"`` — the solver's own always-declared config (controlDict,
      transportProperties, turbulenceProperties, gravity, the mesh dicts). These
      always apply and are **not** listed by :func:`model_catalog`; author the
      physics/time ones for every case (the mesh dicts only when meshing).
    * ``"required_model"`` — owned by a required model family (always active).
    * ``"optional_model"`` — owned by an optional model; author only to enable it.
    """

    name: str
    cls_name: str
    file: str | None = None
    origin: Literal["solver", "required_model", "optional_model"] = "solver"
    description: str | None = None


class ConfigSchema(BaseModel):
    """JSON Schema + rjsf ui-schema + defaults for one config class."""

    name: str
    json_schema: dict[str, Any]
    ui_schema: dict[str, Any]
    defaults: dict[str, Any]


class ToolInfo(BaseModel):
    """One registered preprocessing tool + the config class that writes its dict.

    ``name`` is what goes in a ``PreprocessConfig`` ``tools`` entry (``blockMesh`` /
    ``snappyHexMesh`` / ``checkMesh``); ``step_schema`` is that entry's JSON Schema;
    ``dict_file`` is the case file the tool reads and ``config`` the config class that
    writes it — both ``None`` for a tool that reads no dict (e.g. ``checkMesh``).
    """

    name: str
    description: str | None = None
    step_schema: dict[str, Any]
    dict_file: str | None = None
    config: str | None = None


class ModelSummary(BaseModel):
    """A solver model for a "select models" UI (owned configs as class-name strings)."""

    name: str
    label: str
    required: bool
    dicts: list[str]
    fields: list[str]

    @classmethod
    def from_entry(cls, entry: Any) -> "ModelSummary":
        return cls(
            name=entry.name,
            label=entry.label,
            required=entry.required,
            dicts=[c.__name__ for c in entry.dicts],
            fields=[c.__name__ for c in entry.fields],
        )


def _describe(obj: Any, *, fallback: str | None = None) -> str | None:
    """First line of ``obj``'s docstring (a one-line purpose), else ``fallback``."""
    doc = (obj.__doc__ or "").strip()
    return doc.split("\n", 1)[0].strip() if doc else fallback


def list_configs(solver: Any) -> list[ConfigInfo]:
    """Every config class ``solver`` may consume (snake name/cls_name/file/origin).

    Each entry's ``origin`` classifies where the config comes from — ``"solver"``
    (always-declared, e.g. transportProperties/turbulenceProperties/gravity — **not**
    in :func:`model_catalog`, author them anyway), ``"required_model"`` (always
    active), or ``"optional_model"`` (author only to enable that model). This is the
    surface that tells an agent what it must author for a runnable case: the
    ``model_catalog`` model-owned configs are a *subset* of the required inputs; the
    ``"solver"``-origin physics/time dicts complete them.
    """
    from neofoam.framework.solver.configurations import (
        _snake_case,
        configurations,
        model_catalog as _model_catalog,
    )

    required_owned: set[type] = set()
    optional_owned: set[type] = set()
    for entry in _model_catalog(solver):
        target = required_owned if entry.required else optional_owned
        target.update(entry.dicts)
        target.update(entry.fields)

    def _origin(cls: type) -> Literal["solver", "required_model", "optional_model"]:
        if cls in required_owned:
            return "required_model"
        if cls in optional_owned:
            return "optional_model"
        return "solver"

    out: list[ConfigInfo] = []
    for cls in configurations(solver).classes:
        io = cls.io_config
        file = io.file if io is not None else None
        out.append(
            ConfigInfo(
                name=_snake_case(cls.__name__),
                cls_name=cls.__name__,
                file=file,
                origin=_origin(cls),
                # synthesised field/fv configs carry no docstring; fall back to the file
                description=_describe(
                    cls, fallback=f"Config for {file}" if file else None
                ),
            )
        )
    return out


def config_schema(solver: Any, name: str) -> ConfigSchema:
    """JSON Schema + rjsf ui-schema + defaults for one config class of ``solver``.

    Accepts either the class name (``ControlDictConfig``) or the snake-case name
    (``control_dict_config``) that :func:`list_configs` advertises. The response
    ``name`` is always the canonical snake-case, regardless of the spelling passed.
    """
    from neofoam.framework.solver.configurations import _snake_case, configurations

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
    return ConfigSchema(
        name=_snake_case(cls.__name__),
        json_schema=json_schema,
        ui_schema=rjsf_uischema(json_schema),
        defaults=default_values(cls),
    )


def tool_catalog(solver: Any) -> list[ToolInfo]:
    """Every registered preprocessing tool, its entry schema, and the config it reads.

    ``solver`` supplies the config surface so each tool's ``dict_file`` is linked to
    the config class that writes it (``blockMesh`` → ``BlockMeshDictConfig``).
    """
    from neofoam.framework.solver.configurations import configurations

    import neofoam.tools  # noqa: F401  (import populates the tool registry)
    from neofoam.tools.registry import available_tools

    file_to_config = {
        cls.io_config.file: cls.__name__
        for cls in configurations(solver).classes
        if cls.io_config is not None
    }

    out: list[ToolInfo] = []
    for tool in available_tools():
        step = tool.step_config_type
        dict_file = None
        if step is not None and "dict_file" in step.model_fields:
            default = step.model_fields["dict_file"].default
            dict_file = default if isinstance(default, str) else None
        out.append(
            ToolInfo(
                name=tool.name,
                description=_describe(step),  # the step config's one-line docstring
                step_schema=step.model_json_schema() if step is not None else {},
                dict_file=dict_file,
                config=file_to_config.get(dict_file) if dict_file else None,
            )
        )
    return out


def model_catalog(solver: Any) -> list[ModelSummary]:
    """Every model of ``solver`` with its required flag + owned configs.

    This lists **model-owned** configs only. A solver also has always-declared
    configs (controlDict, transportProperties, turbulenceProperties, gravity) that
    no model owns — these are **not** here; find them in :func:`list_configs` with
    ``origin == "solver"``. Author the model-owned *and* the ``"solver"``-origin
    configs for a runnable case; this catalog alone is not the full required set.
    """
    from neofoam.framework.solver.configurations import model_catalog as _model_catalog

    return [ModelSummary.from_entry(e) for e in _model_catalog(solver)]
