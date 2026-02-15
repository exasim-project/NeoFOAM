# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

from typing import Any

import pybFoam as pyf

from neofoam.framework.context import FieldUpdates

from .base import (
    detect_algorithm_type,
    get_pressure_velocity_algorithm,
    get_pressure_velocity_operation,
)
from . import pimple as _pimple  # noqa: F401
from . import piso as _piso  # noqa: F401
from . import simple as _simple  # noqa: F401


class PressureVelocityModelImpl:
    def _active_algorithm(self, model_state: Any) -> Any:
        return get_pressure_velocity_algorithm(model_state.algorithm_type)

    def detect(self) -> bool:
        return True

    def load(self, model_state: Any) -> None:
        model_state.use_boussinesq = False
        model_state.pRefCell = None
        model_state.pRefValue = None

        fv_solution = pyf.dictionary.read("system/fvSolution")
        model_state.algorithm_type = detect_algorithm_type(fv_solution)
        model_state.fv_solution = fv_solution

    def resolve(self, _config: Any) -> None:
        pass

    def build(self, model_state: Any) -> list[Any]:
        return self._active_algorithm(model_state).build()

    def inner_loop(self, model_state: Any, ctx: Any) -> bool:
        return self._active_algorithm(model_state).inner_loop(ctx)

    def momentum(self, model_state: Any, **kwargs: Any) -> FieldUpdates:
        op = get_pressure_velocity_operation(model_state.algorithm_type, "momentum")
        return op(
            model_state=model_state,
            **kwargs,
        )

    def continuity(self, model_state: Any, **kwargs: Any) -> FieldUpdates:
        op = get_pressure_velocity_operation(model_state.algorithm_type, "continuity")
        return op(
            model_state=model_state,
            **kwargs,
        )
