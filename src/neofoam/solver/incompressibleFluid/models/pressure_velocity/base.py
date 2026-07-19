# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity coupling dispatcher.

Detects which algorithm to instantiate from ``system/fvSolution``.
PIMPLE and SIMPLE (steady state) are wired up; PISO falls back to
PIMPLE rather than producing a missing-algorithm error so existing
tutorial cases that specify it still run.
"""

from typing import Any

import pybFoam as pyf

from .pimpleAlgorithm import pimple
from .simpleAlgorithm import simple


class PressureVelocityAlgorithm:
    """Dispatcher for pressure-velocity coupling algorithms."""

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free (no detection).

        Used to enumerate the family's config classes (per-spec
        ``fvSchemes`` / ``fvSolution`` slices) for the solver schema.
        PISO would join the list as it is ported.
        """
        return [pimple, simple]

    @classmethod
    def detect_and_create(cls) -> Any:
        """Detect algorithm from fvSolution and prime its state."""
        fv_solution = pyf.dictionary.read("system/fvSolution")

        if fv_solution.found("PIMPLE"):
            algorithm_type = "PIMPLE"
        elif fv_solution.found("SIMPLE"):
            algorithm_type = "SIMPLE"
        elif fv_solution.found("PISO"):
            algorithm_type = "PISO"
        else:
            algorithm_type = "PIMPLE"

        algorithm_model = simple if algorithm_type == "SIMPLE" else pimple
        algorithm_model.algorithm_type = algorithm_type  # type: ignore[attr-defined]
        algorithm_model.use_boussinesq = False  # type: ignore[attr-defined]
        algorithm_model.pRefCell = None  # type: ignore[attr-defined]
        algorithm_model.pRefValue = None  # type: ignore[attr-defined]
        return algorithm_model

    @classmethod
    def create(cls, *, algorithm_type: str) -> Any:
        """Programmatically create a pressure-velocity algorithm model."""
        if algorithm_type in {"Simple", "SIMPLE"}:
            simple.algorithm_type = "SIMPLE"  # type: ignore[attr-defined]
            return simple
        if algorithm_type not in {"Pimple", "PIMPLE"}:
            raise ValueError(
                f"Only the PIMPLE and SIMPLE algorithms are wired up; "
                f"requested {algorithm_type!r}."
            )
        pimple.algorithm_type = "PIMPLE"  # type: ignore[attr-defined]
        return pimple
