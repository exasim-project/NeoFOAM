# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pure-Python ``laminar`` momentum-transport on NeoN — a :class:`ModelSpec`.

The NeoN mirror of :mod:`neofoam.turbulence.models.laminar`: ``laminar`` is a
:class:`~neofoam.framework.model.ModelSpec` + config, **no class**. It registers
with the runtime-selectable :class:`~neofoam.turbulence.neon.neonMomentumTransportModel`
family under the ``turbulenceProperties`` name ``laminar``.

Laminar flow has no eddy viscosity, so this owns no transport equation and no
``correct`` operation: its ``@build`` emits ``nut = 0`` and the effective (surface)
viscosity ``nuEff = surfaceInterpolate(nu)`` — assembled entirely from NeoN Python
primitives, no C++ turbulence factory. It reproduces the C++ laminar model's
``nut`` / ``nuEff`` exactly.

It owns no ``grad(U)``: that is a kinematic field of the velocity, computed by the
momentum predictor where the viscous stress consumes it — the turbulence model uses
``grad(U)`` but does not produce it.

This is the template for a pure-Python closure with a transport equation
(:mod:`neofoam.turbulence.models.neon_kEpsilon`): add its transport fields in
``@build`` and solve their PDEs in an ``@operation``.
"""

from typing import Any

import neon._neon as nn  # NeoN surface interpolation
from neofoam import neofoam_bindings as nfb
from neofoam.framework.initialization import InitStep
from neofoam.framework.initialization import field as init_field

from ..config import TurbulencePropertiesConfig
from ..neon import Model, neonMomentumTransportModel

__all__ = ["neon_laminar"]

neon_laminar = Model("laminar").register_with(neonMomentumTransportModel)
neon_laminar.config(TurbulencePropertiesConfig)


@neon_laminar.build
def build(config: TurbulencePropertiesConfig) -> list[InitStep]:
    """Emit the ``nut`` (zero) and ``nuEff`` (surface) fields laminar owns.

    The NeoN ``runtime`` and molecular ``nu`` are injected by name from the
    Context the wrapper seeds (``models.neon_runtime`` / ``models.nu_vol``).
    """

    def create_nut(ctx: dict[str, Any]) -> Any:
        return nfb.create_uniform_volume_field(ctx["models.neon_runtime"], "nut", 0.0)

    def create_nu_eff(ctx: dict[str, Any]) -> Any:
        rt = ctx["models.neon_runtime"]
        surf = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )
        return surf.interpolate(ctx["models.nu_vol"])  # nuEff = surf(nu), nut = 0

    return [
        init_field("nut", create_nut, depends_on=["models.neon_runtime"]),
        init_field(
            "nuEff",
            create_nu_eff,
            depends_on=["models.neon_runtime", "models.nu_vol"],
        ),
    ]
