# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pure dispatcher for pressure-velocity coupling algorithms.

This module detects which algorithm (PIMPLE/SIMPLE/PISO) to use from fvSolution
and instantiates the appropriate algorithm model.
"""

from typing import Any

import pybFoam as pyf

from .pimpleAlgorithm import pimple
from .simpleAlgorithm import simple
from .pisoAlgorithm import piso


class PressureVelocityAlgorithm:
    """Dispatcher for pressure-velocity coupling algorithms."""

    @classmethod
    def detect_and_create(cls) -> Any:
        """
        Detect algorithm from fvSolution and create the appropriate model instance.

        Returns:
            Algorithm model instance (pimple, simple, or piso) with initialized state.
        """
        # Read fvSolution to detect the algorithm
        fv_solution = pyf.dictionary.read("system/fvSolution")
        toc = set(fv_solution.toc())

        # Check which algorithm is specified in fvSolution
        if "PIMPLE" in toc:
            algorithm_model = pimple
            algorithm_type = "PIMPLE"
        elif "SIMPLE" in toc:
            algorithm_model = simple
            algorithm_type = "SIMPLE"
        elif "PISO" in toc:
            algorithm_model = piso
            algorithm_type = "PISO"
        else:
            # Default to PIMPLE
            algorithm_model = pimple
            algorithm_type = "PIMPLE"

        # Initialize state attributes needed by the algorithms
        algorithm_model.use_boussinesq = False
        algorithm_model.pRefCell = None
        algorithm_model.pRefValue = None
        algorithm_model.fv_solution = None
        algorithm_model.algorithm_type = algorithm_type

        # Load the algorithm model if it has a load method
        if hasattr(algorithm_model, "run_load"):
            algorithm_model.run_load()

        return algorithm_model

    @classmethod
    def create(cls, *, algorithm_type: str) -> Any:
        """
        Programmatically create a specific algorithm model.

        Args:
            algorithm_type: One of "Pimple", "Simple", or "Piso"

        Returns:
            Algorithm model instance.
        """
        algorithms = {
            "Pimple": pimple,
            "Simple": simple,
            "Piso": piso,
        }
        if algorithm_type not in algorithms:
            raise ValueError(
                f"Unsupported pressure-velocity algorithm: {algorithm_type}. "
                f"Available: {list(algorithms.keys())}"
            )
        return algorithms[algorithm_type]
