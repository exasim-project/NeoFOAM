# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity coupling dispatcher for the NeoN solver.

Detects which algorithm to instantiate from ``system/fvSolution``.
PIMPLE (transient) and SIMPLE (steady state) are wired up; PISO falls
back to PIMPLE rather than producing a missing-algorithm error, so
existing tutorial cases that specify it still run (the legacy
``neoPimpleFoam`` also reads its corrector counts from the ``PIMPLE``
subdict only).
"""

from dataclasses import dataclass
from typing import Any

import pybFoam as pyf

from neofoam.framework.model import ModelRuntime, ModelSpec

from .pimpleAlgorithm import pimpleNeoN
from .simpleAlgorithm import simpleNeoN


@dataclass(frozen=True)
class PressureVelocitySettingsNeoN:
    """One run's settings, kept off the spec that regions and runs share."""

    algorithm_type: str


def _runtime(spec: ModelSpec, algorithm_type: str) -> ModelRuntime:
    return ModelRuntime(
        spec=spec, name=spec.name, config=PressureVelocitySettingsNeoN(algorithm_type)
    )


class PressureVelocityAlgorithmNeoN:
    """Dispatcher for NeoN pressure-velocity coupling algorithms."""

    @classmethod
    def all_specs(cls) -> list[Any]:
        """Every member spec of the family, case-free (no detection)."""
        return [pimpleNeoN, simpleNeoN]

    @classmethod
    def detect_and_create(cls) -> ModelRuntime:
        """Detect the algorithm from fvSolution: a fresh runtime of its spec."""
        fv_solution = pyf.dictionary.read("system/fvSolution")

        if fv_solution.found("PIMPLE"):
            algorithm_type = "PIMPLE"
        elif fv_solution.found("SIMPLE"):
            algorithm_type = "SIMPLE"
        elif fv_solution.found("PISO"):
            algorithm_type = "PISO"
        else:
            algorithm_type = "PIMPLE"

        return _runtime(simpleNeoN if algorithm_type == "SIMPLE" else pimpleNeoN, algorithm_type)

    @classmethod
    def create(cls, *, algorithm_type: str) -> ModelRuntime:
        """Programmatically create a pressure-velocity algorithm runtime."""
        if algorithm_type in {"Simple", "SIMPLE"}:
            return _runtime(simpleNeoN, "SIMPLE")
        if algorithm_type not in {"Pimple", "PIMPLE"}:
            raise ValueError(
                f"incompressibleFluidNeoN only supports the PIMPLE and SIMPLE "
                f"algorithms; requested {algorithm_type!r}."
            )
        return _runtime(pimpleNeoN, "PIMPLE")
