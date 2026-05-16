# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turbulence model configurations for comparison tests.

Each model defines:
- Which fields to compare after correct()
- The NeoN correct() function and how to call it
- Tolerances
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)
from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
    correct as sa_correct,
)

from generate_fields import compute_nut_from_nuTilda


@dataclass
class TurbulenceModelConfig:
    """Configuration for a turbulence model comparison test."""

    name: str
    comparison_fields: list[str]
    neon_correct: Callable[..., None]
    build_neon_args: Callable[[dict[str, Any]], dict[str, Any]]
    nut_from_primary: Callable[..., np.ndarray]
    rtol: float = 1e-3
    atol: float = 1e-10


def _sa_build_neon_args(env: dict[str, Any]) -> dict[str, Any]:
    return {
        "cfg": SpalartAllmarasConfig(),
        "rt": env["rt"],
        "nuTilda": env["nuTilda"],
        "nut": env["nut"],
        "U": env["U"],
        "phi": env["phi"],
        "d": env["d"],
        "nu_value": env["nu_value"],
    }


SPALART_ALLMARAS = TurbulenceModelConfig(
    name="SpalartAllmaras",
    comparison_fields=["nuTilda", "nut"],
    neon_correct=sa_correct,
    build_neon_args=_sa_build_neon_args,
    nut_from_primary=compute_nut_from_nuTilda,
)

MODELS: dict[str, TurbulenceModelConfig] = {
    "SpalartAllmaras": SPALART_ALLMARAS,
}
