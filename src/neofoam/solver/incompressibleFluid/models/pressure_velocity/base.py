# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pressure-velocity coupling dispatcher.

Detects which algorithm to instantiate from ``system/fvSolution``.
PIMPLE and SIMPLE (steady state) are wired up; PISO falls back to
PIMPLE rather than producing a missing-algorithm error so existing
tutorial cases that specify it still run.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf

from neofoam.framework.model import ModelRuntime, ModelSpec

from .pimpleAlgorithm import pimple
from .simpleAlgorithm import simple


@dataclass
class PressureVelocitySettings:
    """One run's settings, kept off the spec that regions and runs share."""

    algorithm_type: str
    #: Set by the boussinesq model's resolve: dispatches the buoyant variants.
    use_boussinesq: bool = False
    #: The case directory the run reads ``system/fvSolution`` from (a region's, in a
    #: multi-region case).
    case_dir: Path = field(default_factory=lambda: Path("."))


def _runtime(spec: ModelSpec, algorithm_type: str, case_dir: Optional[Path] = None) -> ModelRuntime:
    settings = PressureVelocitySettings(algorithm_type, case_dir=Path(case_dir or "."))
    return ModelRuntime(spec=spec, name=spec.name, config=settings)


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
    def detect_and_create(cls, case_dir: Optional[Path] = None) -> ModelRuntime:
        """Detect the algorithm from ``case_dir``'s fvSolution (default: the working
        directory's): a fresh runtime of its spec, reading its case from there."""
        fv_solution = pyf.dictionary.read(str(Path(case_dir or ".") / "system" / "fvSolution"))

        if fv_solution.found("PIMPLE"):
            algorithm_type = "PIMPLE"
        elif fv_solution.found("SIMPLE"):
            algorithm_type = "SIMPLE"
        elif fv_solution.found("PISO"):
            algorithm_type = "PISO"
        else:
            algorithm_type = "PIMPLE"

        return _runtime(simple if algorithm_type == "SIMPLE" else pimple, algorithm_type, case_dir)

    @classmethod
    def create(cls, *, algorithm_type: str) -> ModelRuntime:
        """Programmatically create a pressure-velocity algorithm runtime."""
        if algorithm_type in {"Simple", "SIMPLE"}:
            return _runtime(simple, "SIMPLE")
        if algorithm_type not in {"Pimple", "PIMPLE"}:
            raise ValueError(
                f"Only the PIMPLE and SIMPLE algorithms are wired up; requested {algorithm_type!r}."
            )
        return _runtime(pimple, "PIMPLE")
