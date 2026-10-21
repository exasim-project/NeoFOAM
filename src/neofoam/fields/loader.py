# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""One-call read/write orchestration for declared ``0/<name>`` fields.

These helpers are the user-facing surface for the per-field schemas
``Model.field(...)`` declares. The schemas themselves are pydantic
``BaseConfig`` subclasses — :func:`load_fields` just walks the solver's
field schemas through :class:`~neofoam.framework.solver.configurations.Configurations.fields`
and calls each class's ``.load(case_dir=...)``, returning a
``dict[str, BaseConfig]`` keyed by field name. :func:`save_fields`
mirrors that on the write side.

Field files are single-owner per ``0/<name>``, so direct ``cfg.save()``
per field is enough — unlike ``constant/transportProperties`` or
``system/fvSchemes`` (multi-owner files), there's no merge layer. The
caller can mutate, omit, or add fields freely between load and save;
:func:`save_fields` writes exactly what it is handed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

from neofoam.framework.solver.configurations import configurations
from neofoam.io.base import BaseConfig


def load_fields(case_dir: Union[Path, str], *, solver: Any) -> dict[str, BaseConfig]:
    """Load every declared ``0/<name>`` field on ``solver`` from ``case_dir``.

    Walks ``configurations(solver).fields`` and calls
    ``cls.load(case_dir=...)`` for each. Missing files raise
    ``FileNotFoundError`` from the underlying strategy — the loader
    does not silently skip absent fields because, by construction,
    every declared field is one the solver expects to be present.

    Args:
        case_dir: Case directory containing the ``0/`` field files.
        solver: A solver spec (e.g. ``incompressibleFluid``) whose
            bound model registry declares fields via ``Model.field``.

    Returns:
        Mapping of field name → loaded :class:`BaseConfig` instance.
        The name is taken from the schema class — ``"UFieldConfig"``
        → ``"U"`` — so callers index by the OpenFOAM field name
        directly.
    """
    case_path = Path(case_dir)
    out: dict[str, BaseConfig] = {}
    for cls in configurations(solver).fields:
        name = _field_name(cls)
        out[name] = cls.load(case_dir=case_path)
    return out


def save_fields(
    fields: Mapping[str, BaseConfig],
    case_dir: Union[Path, str],
) -> list[Path]:
    """Persist a mapping of fields back to ``case_dir``.

    Each instance is written through its registered IO strategy — the
    ``@IOStrategy(OF("0/<name>"))`` decorator that ``schema_for(decl)``
    stamped on the synthesised class. Returns the list of paths
    actually written, in iteration order, so the caller can log /
    surface them in tests.

    Args:
        fields: Field instances to persist. The mapping key is
            informational only — the on-disk path is driven by each
            instance's ``io_config.file``.
        case_dir: Target case directory.

    Returns:
        Paths written, in iteration order.

    Raises:
        ValueError: If an instance's class has no ``io_config`` (it was
            not synthesised by :func:`schema_for` — and there is no
            ``0/<name>`` binding to write to).
    """
    case_path = Path(case_dir)
    written: list[Path] = []
    for _name, instance in fields.items():
        # ``getattr`` (not ``type(instance).io_config``) so plain pydantic
        # models that never went through ``@IOStrategy`` raise the
        # ValueError below rather than ``AttributeError`` — the former is
        # the documented failure mode for this function.
        io = getattr(type(instance), "io_config", None)
        if io is None:
            raise ValueError(
                f"save_fields: {type(instance).__name__} has no io_config "
                "(not a synthesised field schema)"
            )
        instance.save(case_dir=case_path)
        written.append(case_path / io.file)
    return written


def _field_name(cls: type[BaseConfig]) -> str:
    """Recover the field name from a synthesised schema class.

    Synthesised schemas are named ``"<name>FieldConfig"`` in
    :func:`schema_for`; their ``io_config.file`` is ``"0/<name>"``.
    Both are stable, but the path is the canonical source — strip the
    ``0/`` prefix to get the field name even if the class name is later
    changed for cosmetic reasons.
    """
    io = cls.io_config
    if io is not None and io.file.startswith("0/"):
        return io.file[len("0/") :]
    # Fallback to the class name convention; should never fire for
    # schemas built via ``schema_for``.
    return cls.__name__.removesuffix("FieldConfig")
