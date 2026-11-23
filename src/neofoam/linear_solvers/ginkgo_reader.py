# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Read existing OpenFOAM-named and full Ginkgo blocks as the simple Ginkgo form."""

from __future__ import annotations

import warnings
from typing import Any, Optional

from neofoam.linear_solvers.base import (
    _CONTROL_KEYS,
    LinearSolverConversionWarning,
    Preconditioner,
    _plugin_tag,
    _registered_plugins,
    registered_preconditioners,
)

FOAM_TO_GINKGO_SOLVERS = {
    "PCG": "Cg",
    "PBiCG": "Bicg",
    "PBiCGStab": "Bicgstab",
    "smoothSolver": "Ir",
    "GAMG": "Cg",
}
# GAMG replaces the block's preconditioner, so its DIC word never reaches Ginkgo.
_DIC_SOLVERS = (frozenset(FOAM_TO_GINKGO_SOLVERS) - {"GAMG"}) | {"Ginkgo"}
_CRITERIA_CONTROLS = {"absolute_residual_norm": "tolerance", "initial_residual_norm": "relTol"}

_TRUE_WORDS = frozenset({"true", "yes", "on", "y", "t", "1"})
_DEFAULT_SETTINGS: dict[str, frozenset[str]] = {
    "l1ScaledResidual": _TRUE_WORDS,
    "reuseSetup": _TRUE_WORDS,
    "checkFrequency": frozenset({"1"}),
}
# The mapper removes these on every path without using their values.
_MAPPER_IGNORED = frozenset({"smoother", "nSweeps"})
# A multigrid path drops unknown keys, but these are honoured or change how the block runs.
_NEVER_DROPPED = frozenset(
    {
        "backend",
        "checkFrequency",
        "configFile",
        "coupled",
        "criteria",
        "l1ScaledResidual",
        "minIterFactor",
        "reportName",
        "type",
    }
)


def _as_text(value: Any) -> Any:
    """A parsed value with every leaf as text, as the OpenFOAM reader returns it."""
    if isinstance(value, dict):
        return {key: _as_text(item) for key, item in value.items()}
    return str(value)


def _matching_preconditioner(value: Any) -> Optional[type[Preconditioner]]:
    """The Ginkgo plugin owning a mapper word or a full Ginkgo dictionary."""
    text = _as_text(value)
    for plugin in registered_preconditioners("ginkgo"):
        if text in (plugin.mapper_word, plugin.native_block):
            return plugin
    return None


def _named_preconditioner(block: dict[str, Any], solver: str) -> type[Preconditioner]:
    """The preconditioner plugin a block selects; only ``Ir`` has a mapper default."""
    value = block.get("preconditioner")
    if value is None and solver == "Ir":
        return _plugin_named("Jacobi")
    named = None if value is None else _matching_preconditioner(value)
    if named is None:
        words = sorted(plugin.mapper_word for plugin in registered_preconditioners("ginkgo"))
        raise ValueError(
            f"preconditioner {value!r} of {block.get('solver')} has no simple equivalent; "
            f"use one of {words} or a Ginkgo configFile"
        )
    return named


def _plugin_named(tag: str) -> type[Preconditioner]:
    """The Ginkgo preconditioner plugin with this ``type``."""
    plugins = registered_preconditioners("ginkgo")
    return next(plugin for plugin in plugins if _plugin_tag(plugin, "type") == tag)


def _keeps_default(key: str, value: Any) -> bool:
    """True when the setting has the value NeoN uses without it."""
    return str(value).lower() in _DEFAULT_SETTINGS.get(key, frozenset())


def _dropped_keys(block: dict[str, Any], known: set[str], drops_unknown: bool) -> list[str]:
    """The keys safe to drop; raises for a key whose value NeoN would honour or fail on."""
    dropped: list[str] = []
    for key, value in block.items():
        if key in known or _keeps_default(key, value):
            continue
        if key not in _MAPPER_IGNORED and (not drops_unknown or key in _NEVER_DROPPED):
            raise ValueError(
                f"{key} {value!r} of {block.get('solver')} has no simple equivalent; "
                "use a Ginkgo configFile"
            )
        dropped.append(key)
    return dropped


def _simple_block(
    block: dict[str, Any],
    solver: str,
    preconditioner: type[Preconditioner],
    known: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The simple form of a parsed block, warning about the keys it drops."""
    option_keys = preconditioner.foam_options
    options = {key: block[key] for key in option_keys if key in block}
    selected = {"type": _plugin_tag(preconditioner, "type"), **options}
    simple: dict[str, Any] = {"solver": solver, "preconditioner": selected}
    simple |= {key: block[key] for key in _CONTROL_KEYS if key in block}
    handled = {"solver", "preconditioner", *_CONTROL_KEYS, *option_keys, *known}
    dropped = _dropped_keys(block, handled, preconditioner.drops_unknown_keys)
    if block.get("solver") == "GAMG" and "preconditioner" in block:
        dropped.append("preconditioner")
    if dropped:
        warnings.warn(
            f"{block.get('solver')} converted to {solver}: dropped {sorted(dropped)}, "
            "which NeoN ignores",
            LinearSolverConversionWarning,
            stacklevel=2,
        )
    return simple


def _from_openfoam(block: dict[str, Any]) -> dict[str, Any]:
    """The Ginkgo equivalent of an OpenFOAM-named block."""
    solver = FOAM_TO_GINKGO_SOLVERS[block["solver"]]
    if block["solver"] == "GAMG":
        return _simple_block(block, solver, _matching_preconditioner("GAMG"))  # type: ignore[arg-type]
    return _simple_block(block, solver, _named_preconditioner(block, solver))


def _with_criteria_controls(block: dict[str, Any]) -> dict[str, Any]:
    """Move the stopping norms to their controls; top-level keys win, as in the mapper."""
    criteria = block.get("criteria", {})
    unsupported = sorted(set(criteria) - set(_CRITERIA_CONTROLS))
    if unsupported:
        raise ValueError(f"criteria {unsupported} have no simple equivalent; use a configFile")
    moved = {_CRITERIA_CONTROLS[key]: value for key, value in criteria.items()}
    return {**moved, **{key: value for key, value in block.items() if key != "criteria"}}


def _from_native(block: dict[str, Any]) -> dict[str, Any]:
    """The simple form of a full ``solver Ginkgo; type solver::X`` block."""
    names = [_plugin_tag(plugin) for plugin in _registered_plugins("ginkgo")]
    solver_type = str(block.get("type"))
    solver = solver_type.removeprefix("solver::")
    if not solver_type.startswith("solver::") or solver not in names:
        raise ValueError(
            f"Ginkgo type {block.get('type')!r} has no simple form; supported: "
            f"{[f'solver::{name}' for name in names if name != 'Ginkgo']}, or use a configFile"
        )
    controls = _with_criteria_controls(block)
    return _simple_block(
        controls, solver, _named_preconditioner(controls, solver), frozenset({"type"})
    )


def _read_ginkgo_block(block: Any) -> Any:
    """Convert an existing OpenFOAM or full Ginkgo block to the simple form."""
    if not isinstance(block, dict):
        return block
    name = block.get("solver")
    if name in FOAM_TO_GINKGO_SOLVERS:
        return _from_openfoam(block)
    if name == "Ginkgo" and "configFile" not in block:
        return _from_native(block)
    if name not in {_plugin_tag(plugin) for plugin in _registered_plugins("ginkgo")}:
        raise ValueError(f"unsupported linear-solver block {block!r}")
    return block


def _converted_pressure_block(block: Any) -> Any:
    """A ``p`` block with OpenFOAM's ``DIC`` as Jacobi and a missing choice set to Jacobi."""
    if not isinstance(block, dict) or "configFile" in block:
        return block
    name = block.get("solver")
    if block.get("preconditioner") == "DIC" and name in _DIC_SOLVERS:
        warnings.warn(
            f"{name} with DIC on pressure converted to Jacobi: Ginkgo Ic can abort on "
            "the NeoN pressure matrix",
            LinearSolverConversionWarning,
            stacklevel=2,
        )
        return {**block, "preconditioner": "diagonal"}
    if "preconditioner" not in block and name in _preconditioned_ginkgo_tags():
        return {**block, "preconditioner": "Jacobi"}
    return block


def _preconditioned_ginkgo_tags() -> set[str]:
    """The simple Ginkgo solvers that take a preconditioner."""
    plugins = _registered_plugins("ginkgo")
    return {_plugin_tag(plugin) for plugin in plugins if "preconditioner" in plugin.model_fields}
