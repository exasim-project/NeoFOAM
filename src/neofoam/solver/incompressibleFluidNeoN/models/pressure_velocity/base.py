# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity coupling dispatcher for the NeoN solver.

Detects which algorithm to instantiate from ``system/fvSolution``. Only
PIMPLE is supported — SIMPLE / PISO fall back to PIMPLE rather than
producing a missing-algorithm error, so existing tutorial cases that
specify them still run (the legacy ``neoPimpleFoam`` also reads its
corrector counts from the ``PIMPLE`` subdict only).
"""

from typing import Any

import pybFoam as pyf

from .pimpleAlgorithm import pimpleNeoN


class PressureVelocityAlgorithmNeoN:
    """Dispatcher for NeoN pressure-velocity coupling algorithms."""

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free (no detection)."""
        return [pimpleNeoN]

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

        algorithm_model = pimpleNeoN
        algorithm_model.algorithm_type = algorithm_type  # type: ignore[attr-defined]
        return algorithm_model

    @classmethod
    def create(cls, *, algorithm_type: str) -> Any:
        """Programmatically create the PIMPLE algorithm model."""
        if algorithm_type not in {"Pimple", "PIMPLE"}:
            raise ValueError(
                f"incompressibleFluidNeoN only supports the PIMPLE algorithm; "
                f"requested {algorithm_type!r}."
            )
        pimpleNeoN.algorithm_type = "PIMPLE"  # type: ignore[attr-defined]
        return pimpleNeoN
