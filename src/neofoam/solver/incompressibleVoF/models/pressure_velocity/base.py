# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Pressure-velocity coupling dispatcher for incompressibleVoF solver.

VoF solvers always use PIMPLE (interFoam does not support SIMPLE or PISO).
This module detects the PIMPLE dict from fvSolution and initialises the pimple
algorithm model with VoF-aware state attributes.
"""

from typing import Any

import pybFoam as pyf

from .pimpleAlgorithm import pimple


class PressureVelocityAlgorithm:
    """Dispatcher for pressure-velocity coupling - VoF always uses PIMPLE."""

    @classmethod
    def detect_and_create(cls) -> Any:
        """
        Detect algorithm from fvSolution and create the pimple model instance.

        Returns:
            pimple model instance with initialised state attributes.
        """
        fv_solution = pyf.dictionary.read("system/fvSolution")
        toc = set(fv_solution.toc())

        if "PIMPLE" not in toc:
            import warnings

            warnings.warn(
                "No PIMPLE dict found in fvSolution. "
                "incompressibleVoF always uses PIMPLE. "
                "Continuing with default PIMPLE settings.",
                stacklevel=2,
            )

        algorithm_model = pimple
        algorithm_model.algorithm_type = "PIMPLE"

        if hasattr(algorithm_model, "run_load"):
            algorithm_model.run_load()

        return algorithm_model
