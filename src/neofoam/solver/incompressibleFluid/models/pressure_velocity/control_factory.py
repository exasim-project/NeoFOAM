# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Factory helpers for native NeoFOAM pressure-velocity controls."""

from typing import Any

import pybFoam as pyf

from neofoam.algorithms.control import PimpleControl, SimpleControl


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


def _dict_read_required(dictionary: Any, key: str, py_type: type) -> Any:
    """Read a required value from the dictionary; raise KeyError if missing."""
    if dictionary is None or not hasattr(dictionary, "found"):
        raise KeyError(f"'{key}' not found: dictionary is not available")
    if not dictionary.found(key):
        raise KeyError(f"Required key '{key}' not found in fvSolution dictionary")
    return dictionary.get[py_type](key)


def _dict_read_value(dictionary: Any, key: str, default: Any) -> Any:
    """Read an optional value from the dictionary, returning default if missing."""
    if dictionary is None:
        return default

    if not (hasattr(dictionary, "getOrDefault") and hasattr(dictionary, "found")):
        return default

    if not dictionary.found(key):
        return default

    try:
        proxy = dictionary.getOrDefault
        if isinstance(default, bool):
            return proxy[bool](key, default)
        elif isinstance(default, int):
            return proxy[int](key, default)
        elif isinstance(default, float):
            return proxy[float](key, default)
        elif isinstance(default, str):
            return proxy[str](key, default)
        else:
            return proxy[int](key, default)
    except Exception:
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
        nOuterCorrectors=int(_dict_read_value(pimple_dict, "nOuterCorrectors", 1)),
        nCorrectors=int(_dict_read_required(pimple_dict, "nCorrectors", int)),
        nNonOrthogonalCorrectors=int(
            _dict_read_required(pimple_dict, "nNonOrthogonalCorrectors", int)
        ),
        momentumPredictor=_coerce_bool(
            _dict_read_value(pimple_dict, "momentumPredictor", True), True
        ),
        turbCorr=_coerce_bool(_dict_read_value(pimple_dict, "turbCorr", True), True),
    )


def create_simple_control(_context: dict[str, Any]) -> SimpleControl:
    """Create native `SimpleControl` from fvSolution settings."""
    simple_dict = _read_algorithm_dict("SIMPLE")

    return SimpleControl(
        nNonOrthogonalCorrectors=_coerce_int(
            _dict_read_value(simple_dict, "nNonOrthogonalCorrectors", 0), 0
        ),
        momentumPredictor=_coerce_bool(
            _dict_read_value(simple_dict, "momentumPredictor", True), True
        ),
        consistent=_coerce_bool(
            _dict_read_value(simple_dict, "consistent", False), False
        ),
        useResidualConvergence=False,
    )
