# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""NeoN laminar (no turbulence) model.

Provides nuEff_surface = nu (laminar viscosity only).
correct() is a no-op.
"""

from typing import Any, Literal

from pydantic import BaseModel

from neofoam import neofoam_bindings as nfb

from neofoam.framework.context import Context, FieldUpdates
from neofoam.framework.initialization import field

from .base import NeonTurbulenceModel


@NeonTurbulenceModel.register
class NeonLaminar(BaseModel):
    """Laminar model — no turbulence, nuEff = nu."""

    turbulence_type: Literal["laminar"] = "laminar"
    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def detect_model() -> bool:
        return False  # Laminar is the fallback, not auto-detected

    def build_steps(self) -> list[Any]:
        return [
            field(
                "nuEff_surface",
                _create_nuEff_surface,
                depends_on=["models.neon_runtime", "models.nu_laminar_value"],
            ),
        ]

    def correct(self, ctx: Context) -> FieldUpdates:
        return FieldUpdates({})


def _create_nuEff_surface(context: dict[str, Any]) -> Any:
    """Create effective viscosity surface field = laminar nu."""
    rt: Any = context["models.neon_runtime"]
    nu_value: float = context["models.nu_laminar_value"]
    return nfb.create_uniform_surface_field(rt, "nuEff", nu_value)
