# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Dispatcher for NeoN-based pressure-velocity coupling (PIMPLE or PISO)."""

from typing import Any

import pybFoam as pyf

from .pisoAlgorithm import piso
from .pimpleAlgorithm import pimple


class PressureVelocityAlgorithm:
    """Dispatcher for NeoN-backed pressure-velocity coupling algorithms."""

    @classmethod
    def detect_and_create(cls) -> Any:
        """Detect algorithm from fvSolution and create the model instance."""
        fv_solution = pyf.dictionary.read("system/fvSolution")

        if fv_solution.found("PIMPLE"):
            algorithm_model = pimple
            algorithm_type = "PIMPLE"
        elif fv_solution.found("PISO"):
            algorithm_model = piso
            algorithm_type = "PISO"
        else:
            raise ValueError(
                "incompressibleFluidNeon supports PIMPLE or PISO. "
                "No PIMPLE or PISO section found in system/fvSolution."
            )

        # Initialize state attributes
        algorithm_model.algorithm_type = algorithm_type  # type: ignore[attr-defined]
        algorithm_model.pRefCell = None  # type: ignore[attr-defined]
        algorithm_model.pRefValue = None  # type: ignore[attr-defined]

        if hasattr(algorithm_model, "run_load"):
            algorithm_model.run_load()

        return algorithm_model
