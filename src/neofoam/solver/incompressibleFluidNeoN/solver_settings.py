# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run-start translation of the typed NeoN ``fvSolution`` blocks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from neofoam.foam.typed_fv_solution import (
    _PRESSURE_FIELDS,
    _TYPED_FIELDS,
    _VECTOR_FIELDS,
    neonFvSolution,
)
from neofoam.linear_solvers import linear_solver_entry
from neofoam.linear_solvers.base import _plugin_tag, _registered_plugins
from neofoam.linear_solvers.ginkgo import _reject_vector_gmres
from neofoam.linear_solvers.ginkgo_reader import _converted_pressure_block


def _matching_key(solvers: Any, field: str) -> Optional[str]:
    """The ``solvers`` key selecting ``field``; several matches are left to the mapper's error."""
    if solvers.contains(field):
        return str(field)
    quoted = [key for key in solvers.keys() if len(key) > 1 and key[0] == key[-1] == '"']
    matches = [key for key in quoted if re.fullmatch(key[1:-1], field)]
    return matches[0] if len(matches) == 1 else None


def _read_leaf(settings: Any, key: str) -> Any:
    """One word, number or sub-dictionary of a NeoN dictionary."""
    for getter in (settings.get_string, settings.get_int, settings.get_double, settings.get_bool):
        try:
            return getter(key)
        except (RuntimeError, TypeError, ValueError):
            continue
    try:
        return _read_block(settings.subDict(key))
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"linear-solver entry {key!r} cannot be read") from error


def _read_config_path(settings: Any) -> str:
    """The ``configFile`` value; NeoN holds a path with ``/`` as a list of tokens."""
    try:
        return str(settings.get_string("configFile"))
    except (RuntimeError, TypeError, ValueError):
        tokens = settings.get_token_list("configFile")
        return "/".join(tokens.get_string(index) for index in range(tokens.size()))


def _read_block(settings: Any) -> dict[str, Any]:
    """A NeoN dictionary as plain values."""
    return {
        key: _read_config_path(settings) if key == "configFile" else _read_leaf(settings, key)
        for key in settings.keys()
    }


def _insert_settings(target: Any, values: dict[str, Any]) -> None:
    """Fill a NeoN dictionary, keeping each value's type."""
    inserters = {
        str: target.insert_string,
        int: target.insert_int,
        float: target.insert_double,
        bool: target.insert_bool,
        dict: lambda key, value: target.insert_dict(key, _settings_dict(target, value)),
    }
    for key, value in values.items():
        inserters[type(value)](key, value)


def _settings_dict(target: Any, values: dict[str, Any]) -> Any:
    """A new NeoN dictionary of ``target``'s type holding ``values``."""
    nested = type(target)()
    _insert_settings(nested, values)
    return nested


def _is_simple_block(settings: Any, tags: set[str]) -> bool:
    """True for a block in the simple form or a ``configFile`` fallback."""
    name = settings.get_string("solver") if settings.contains("solver") else None
    if name == "Ginkgo":
        return bool(settings.contains("configFile"))
    return name in tags


def _pressure_probe(settings: Any) -> dict[str, Any]:
    """The keys of a NeoN block that decide its pressure conversion."""
    probe = {
        key: _read_leaf(settings, key)
        for key in ("solver", "preconditioner")
        if settings.contains(key)
    }
    return {**probe, "configFile": ""} if settings.contains("configFile") else probe


def _convert_pressure_blocks(solvers: Any) -> None:
    """Give ``p``/``pFinal`` their own entry when the pressure rule changes their block.

    A literal key wins over a regex in the mapper, so a regex shared with another field
    keeps that field's settings.
    """
    for field in _PRESSURE_FIELDS:
        key = _matching_key(solvers, field)
        if key is None:
            continue
        probe = _pressure_probe(solvers.subDict(key))
        converted = _converted_pressure_block(probe)
        if converted is probe:
            continue
        settings = type(solvers)(solvers.subDict(key))
        if settings.contains("preconditioner"):
            settings.remove("preconditioner")
        settings.insert_string("preconditioner", converted["preconditioner"])
        solvers.insert_dict(field, settings)


def build_neon_solver_settings(solvers: Any, case_dir: Path) -> None:
    """Build the in-memory backend settings of the simple ``p``/``U`` blocks, in place.

    Call it once at NeoN solver start, before ``map_solver_settings``. OpenFOAM's ``DIC``
    on ``p``/``pFinal`` becomes Jacobi with a warning; other OpenFOAM-named and full Ginkgo
    blocks are left to the mapper unchanged. The file is not touched and no separate
    settings file is generated. Not thread-safe: it mutates ``solvers``.

    Example::

        build_neon_solver_settings(runtime.fv_solution_dict.subDict("solvers"), case_dir)
    """
    _convert_pressure_blocks(solvers)
    entry = linear_solver_entry(neonFvSolution._backend)
    tags = {_plugin_tag(plugin) for plugin in _registered_plugins(neonFvSolution._backend)}
    for field in _TYPED_FIELDS:
        key = _matching_key(solvers, field)
        if key is None or not _is_simple_block(solvers.subDict(key), tags):
            continue
        block = entry.model_validate(_read_block(solvers.subDict(key)))
        if field in _VECTOR_FIELDS:
            _reject_vector_gmres(block)
        settings = type(solvers)()
        _insert_settings(settings, block.root.runtime_settings(case_dir))
        solvers.insert_dict(key, settings)
