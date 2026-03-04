# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Simplified dispatcher for NeoN-based pressure-velocity coupling (PISO only)."""

from typing import Any

import pybFoam as pyf

from .pisoAlgorithm import piso


class PressureVelocityAlgorithm:
    """Dispatcher for NeoN-backed pressure-velocity coupling algorithms."""

    @classmethod
    def detect_and_create(cls) -> Any:
        """Detect PISO algorithm from fvSolution and create the model instance."""
        fv_solution = pyf.dictionary.read("system/fvSolution")

        if not fv_solution.found("PISO"):
            raise ValueError(
                "incompressibleFluidNeon currently only supports PISO. "
                "No PISO section found in system/fvSolution."
            )

        algorithm_model = piso

        # Initialize state attributes
        algorithm_model.algorithm_type = "PISO"  # type: ignore[attr-defined]
        algorithm_model.pRefCell = None  # type: ignore[attr-defined]
        algorithm_model.pRefValue = None  # type: ignore[attr-defined]

        if hasattr(algorithm_model, "run_load"):
            algorithm_model.run_load()

        return algorithm_model
