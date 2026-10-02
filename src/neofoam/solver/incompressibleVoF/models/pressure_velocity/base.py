# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity coupling dispatcher for incompressibleVoF solver.

VoF solvers always use PIMPLE (interFoam does not support SIMPLE or PISO).
This module detects the PIMPLE dict from ``system/fvSolution`` and primes the
PIMPLE ``ModelSpec`` with VoF-aware state attributes. Ported to the
SolverSpec/ModelSpec API in ``stack/python_arch``.
"""

import warnings
from typing import Any

import pybFoam as pyf

from .pimpleAlgorithm import pimple


class PressureVelocityAlgorithm:
    """Dispatcher for pressure-velocity coupling — VoF always uses PIMPLE."""

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free (no detection).

        VoF wires up only PIMPLE; the entry keeps the family interface
        symmetric with the incompressibleFluid dispatcher.
        """
        return [pimple]

    @classmethod
    def detect_and_create(cls) -> Any:
        """Detect the algorithm from fvSolution and return the PIMPLE spec."""
        fv_solution = pyf.dictionary.read("system/fvSolution")
        if not fv_solution.found("PIMPLE"):
            warnings.warn(
                "No PIMPLE dict found in fvSolution. incompressibleVoF always "
                "uses PIMPLE. Continuing with default PIMPLE settings.",
                stacklevel=2,
            )

        return pimple
