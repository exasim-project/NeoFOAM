# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Read and write a solver's whole case as one aggregate ``CaseSpec`` pydantic model.

Use it to load every config of a case at once (the MCP tools, the case wizard and the
case-fill agent of ``neofoam_ui``); to load or save one config, use its own
:meth:`~neofoam.io.BaseConfig.load` / :func:`neofoam.io.write_configs`.

- :func:`build_case_output_model` — aggregate ``CaseSpec`` pydantic model,
  one ``Optional`` field per ``BaseConfig`` the solver may consume. Suitable
  as a pydantic-ai ``output_type``.
- :func:`load_case_from_disk` — read each config from a source case via the
  registered IO strategy.
- :func:`save_case` — write every non-None field of a ``CaseSpec`` instance
  to the target case via ``write_configs``.
- :func:`read_case_text` — every config-bound file of a case as raw text.

Example::

    spec = load_case_from_disk("pitzDaily")
    save_case(spec, "pitzDaily_copy")
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, Union

import pybFoam as pyf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from neofoam.framework.solver.configurations import _snake_case, configurations
from neofoam.io import BaseConfig, write_configs
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid

#: A key's address in a dict file, from the file's root.
_KeyPath = tuple[str, ...]

__all__ = [
    "build_case_output_model",
    "case_spec_to_configs",
    "load_case_from_disk",
    "read_case_text",
    "save_case",
]


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
