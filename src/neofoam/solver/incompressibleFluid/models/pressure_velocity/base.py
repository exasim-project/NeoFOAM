# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from typing import Any, Callable

import pybFoam as pyf


_pressure_velocity_registry: dict[str, dict[str, Any]] = {}


def register_pressure_velocity_algorithm(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        instance = cls()
        _pressure_velocity_registry[name] = {
            "algorithm": instance,
            "operations": {
                "momentum": instance.momentum,
                "continuity": instance.continuity,
            },
        }
        return cls

    return decorator


def get_pressure_velocity_algorithm(name: str) -> Any:
    if name not in _pressure_velocity_registry:
        raise ValueError(f"Unsupported pressure-velocity algorithm: {name}")
    return _pressure_velocity_registry[name]["algorithm"]


def get_pressure_velocity_operation(name: str, operation: str) -> Any:
    if name not in _pressure_velocity_registry:
        raise ValueError(f"Unsupported pressure-velocity algorithm: {name}")
    operations = _pressure_velocity_registry[name]["operations"]
    if operation not in operations:
        raise ValueError(
            f"Unsupported operation '{operation}' for pressure-velocity algorithm '{name}'"
        )
    return operations[operation]


def detect_algorithm_type(fv_solution: Any) -> str:
    toc = fv_solution.toc()
    if "PIMPLE" in toc:
        return "PIMPLE"
    if "PISO" in toc:
        return "PISO"
    if "SIMPLE" in toc:
        return "SIMPLE"
    raise ValueError("No supported algorithm found in system/fvSolution")


def ensure_pressure_reference(model: Any, p: Any, mesh: Any, p_rgh: Any = None) -> None:
    if model.pRefCell is not None and model.pRefValue is not None:
        return

    if model.fv_solution is None:
        model.fv_solution = pyf.dictionary.read("system/fvSolution")

    algo_dict = model.fv_solution.subDict(model.algorithm_type)
    pressure_field = p_rgh if p_rgh is not None else p
    field_name = "p_rgh" if p_rgh is not None else "p"

    if not (
        algo_dict.found(f"{field_name}RefCell")
        or algo_dict.found(f"{field_name}RefPoint")
    ):
        if p_rgh is not None and (
            algo_dict.found("pRefCell") or algo_dict.found("pRefPoint")
        ):
            pRefCell, pRefValue = pyf.setRefCell(p, algo_dict, True)
        else:
            pRefCell, pRefValue = pyf.setRefCell(pressure_field, algo_dict)
    else:
        pRefCell, pRefValue = pyf.setRefCell(pressure_field, algo_dict)

    model.pRefCell = pRefCell
    model.pRefValue = pRefValue

    mesh.setFluxRequired(pyf.Word("p"))
    if p_rgh is not None:
        mesh.setFluxRequired(pyf.Word("p_rgh"))
