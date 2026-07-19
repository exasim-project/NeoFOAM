# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``laminar`` momentum-transport model — native NeoN **and** pybFoam fallback.

``laminar`` is a :class:`~neofoam.framework.model.ModelSpec` + config, **no
class**, registered with the single :class:`momentumTransportModel` family under
the ``turbulenceProperties`` name ``laminar``. It is the minimal dual-shape model:
it declares both backends in one file, and the consuming solver's ``fallback``
flag picks which runs.

* **native NeoN** (``incompressibleFluidNeoN``, ``fallback=False``): the
  ``@build`` emits ``nut = 0`` and the effective surface viscosity
  ``nuEff = surfaceInterpolate(nu)`` from NeoN primitives — no C++ turbulence
  factory. Laminar flow has no eddy viscosity, so there is no transport equation
  and **no native ``@operation``**; the NeoN ``correct`` is a no-op.
* **pybFoam fallback** (``incompressibleFluid``, ``fallback=True``): the single
  ``fallback=True`` ``correct`` op advances the wrapped pybFoam
  ``incompressibleTurbulenceModel`` (a laminar model: ``nut = 0``, ``nuEff = nu``),
  which owns its own momentum stress.

It owns no ``grad(U)``: that is a kinematic field of the velocity, computed by the
momentum predictor where the viscous stress consumes it.

This is the template for a pure-Python closure with a transport equation
(:mod:`neofoam.turbulence.models.kEpsilon`): add its transport fields in
``@build`` and solve their PDEs in native ``@operation``s.
"""

from typing import Annotated, Any

import neon._neon as nn  # NeoN surface interpolation
from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import InitStep
from neofoam.framework.initialization import field as init_field

from ..config import TurbulencePropertiesConfig
from ..momentumTransport import Model, momentumTransportModel

__all__ = ["laminar"]

laminar = Model("laminar").register_with(momentumTransportModel)
laminar.config(TurbulencePropertiesConfig)


@laminar.build
def build(config: TurbulencePropertiesConfig) -> list[InitStep]:
    """Emit the NeoN ``nut`` (zero) and ``nuEff`` (surface) fields laminar owns.

    The NeoN ``runtime`` and molecular ``nu`` are injected by name from the
    Context the native handle seeds (``models.neon_runtime`` / ``models.nu_vol``).
    Run only on the native path — the fallback path skips ``@build`` entirely.
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


@laminar.operation(name="laminarCorrect", fallback=True)
def correct(
    self: Any,
    turbulence: Annotated[Any, "models"],  # the wrapped pybFoam handle
) -> FieldUpdates:
    """Advance the pybFoam laminar model (``nut = 0``, ``nuEff = nu``).

    Scheduled only on the fallback path (``incompressibleFluid``). The handle is
    resolved from the Context — never captured in the closure — so it stays out
    of the execution-graph reference cycle (see [[project_pybfoam_op_closure_cycle]]).
    """
    turbulence.correct()
    return FieldUpdates({})
