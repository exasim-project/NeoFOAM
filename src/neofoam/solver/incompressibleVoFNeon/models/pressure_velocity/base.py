# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity coupling dispatcher for the NeoN VoF solver.

VoF solvers always use PIMPLE (interFoam does not support SIMPLE or PISO). This
module detects the PIMPLE dict from ``system/fvSolution`` and primes the PIMPLE
``ModelSpec``; a missing PIMPLE dict warns and continues with defaults (mirrors
the pybFoam ``incompressibleVoF`` dispatcher and the legacy ``neoInterFoam``,
which reads its corrector counts from the ``PIMPLE`` subdict only).
"""

import warnings
from typing import Any

import pybFoam as pyf

from .pimpleAlgorithm import pimpleNeoN


class PressureVelocityAlgorithmNeoN:
    """Dispatcher for NeoN VoF pressure-velocity coupling — always PIMPLE."""

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free (no detection)."""
        return [pimpleNeoN]

    @classmethod
    def detect_and_create(cls) -> Any:
        """Detect the algorithm from fvSolution and prime the PIMPLE spec."""
        fv_solution = pyf.dictionary.read("system/fvSolution")
        if not fv_solution.found("PIMPLE"):
            warnings.warn(
                "No PIMPLE dict found in fvSolution. incompressibleVoFNeon always "
                "uses PIMPLE. Continuing with default PIMPLE settings.",
                stacklevel=2,
            )
        pimpleNeoN.algorithm_type = "PIMPLE"  # type: ignore[attr-defined]
        return pimpleNeoN

    @classmethod
    def create(cls, *, algorithm_type: str) -> Any:
        """Programmatically create the PIMPLE algorithm model."""
        if algorithm_type not in {"Pimple", "PIMPLE"}:
            raise ValueError(
                f"incompressibleVoFNeon only supports the PIMPLE algorithm; "
                f"requested {algorithm_type!r}."
            )
        pimpleNeoN.algorithm_type = "PIMPLE"  # type: ignore[attr-defined]
        return pimpleNeoN
