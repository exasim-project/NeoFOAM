# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Factory helpers for native NeoFOAM pressure-velocity controls (VoF solver)."""

from typing import Any

import pybFoam as pyf

from neofoam.algorithms.control import PimpleControl


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "on", "1"}


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dict_read_value(dictionary: Any, key: str, default: Any) -> Any:
    """Read a dictionary value with fallbacks across pybFoam binding variants."""
    if dictionary is None:
        return default

    if hasattr(dictionary, "getOrDefault"):
        try:
            return dictionary.getOrDefault(key, default)
        except Exception:
            pass

    if hasattr(dictionary, "lookupOrDefault"):
        try:
            return dictionary.lookupOrDefault(key, default)
        except Exception:
            pass

    try:
        if hasattr(dictionary, "found") and dictionary.found(key):
            if hasattr(dictionary, "lookup"):
                return dictionary.lookup(key)
    except Exception:
        pass

    return default


def _read_algorithm_dict(algorithm_name: str) -> Any:
    fv_solution = pyf.dictionary.read("system/fvSolution")
    if not hasattr(fv_solution, "subDict"):
        return None
    return fv_solution.subDict(algorithm_name)


def create_pimple_control(_context: dict[str, Any]) -> PimpleControl:
    """Create native `PimpleControl` from fvSolution settings."""
    pimple_dict = _read_algorithm_dict("PIMPLE")

    return PimpleControl(
        nOuterCorrectors=_coerce_int(
            _dict_read_value(pimple_dict, "nOuterCorrectors", 1), 1
        ),
        nCorrectors=_coerce_int(_dict_read_value(pimple_dict, "nCorrectors", 1), 1),
        nNonOrthogonalCorrectors=_coerce_int(
            _dict_read_value(pimple_dict, "nNonOrthogonalCorrectors", 0), 0
        ),
        momentumPredictor=_coerce_bool(
            _dict_read_value(pimple_dict, "momentumPredictor", True), True
        ),
        turbCorr=_coerce_bool(_dict_read_value(pimple_dict, "turbCorr", True), True),
    )
