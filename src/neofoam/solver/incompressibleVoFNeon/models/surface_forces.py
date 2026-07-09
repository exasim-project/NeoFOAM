# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NOTE: no `from __future__ import annotations` — keep annotations live so the
# contributions' field / `Annotated[..., "models"]` params resolve by name.

"""surfaceForces — composable interface forces for incompressibleVoFNeon.

The ``surfaceForces`` model owns the ``interfaceForce`` extension point: the
fold SUMS face-force-density contributions in registration order (empty ->
``None``, "no interface force"). The solver core always wires the two defaults
— surface tension FIRST, gravity (buoyancy) SECOND — so the folded sum
bitwise-reproduces the legacy inline ``fSigma + (-1.0*ghf)*snGrad(rho)``
addition order.

Contract: each contribution returns a face-force *density* (a NeoN scalar
surface field, per unit face area). ``momentum`` consumes the fold as
``reconstruct((F - snGrad(p_rgh)) * magSf)``; ``continuity`` as
``phig = F * rAUf * magSf``. A future force model folds in by decorating a
function with ``@<model>.contributes(interfaceForce)`` and being active for
the case — ``create_fields`` passes the case's optional-model runtimes as bind
candidates, so no solver edit is needed.

These specs are always-on core physics: they are NOT registered with the
``incompressibleVoFNeonModel`` optional family (no case detection, no UI
catalog entry).
"""

from pathlib import Path
from typing import Annotated, Any, Iterable, Optional

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

from .incompressibleVoFNeonModel import Model


def _no_config(_case_dir: Path, _instance_id: Optional[str]) -> None:
    """These models read nothing from the case; ``@load`` lets ``instantiate`` work."""
    return None


surfaceForces = Model("surfaceForces")
surfaceForces.load(_no_config)


@surfaceForces.interface
def interfaceForce(forces: Iterable[Any]) -> Any:
    """Sum the face-force-density contributions in registration order.

    Empty -> ``None`` ("no interface force"); the solver treats that as a
    wiring error since the two defaults are always bound.
    """
    total: Any = None
    for force in forces:
        total = force if total is None else total + force
    return total


# Default contributors. REGISTRATION ORDER IS LOAD-BEARING: surface tension
# first, gravity second — the fold's left-to-right sum then matches the legacy
# `fSigma + (-1.0*ghf)*snGrad(rho)` floating-point addition order bitwise.

surfaceTensionForce = Model("surfaceTensionForce")
surfaceTensionForce.load(_no_config)


@surfaceTensionForce.contributes(interfaceForce)
def surface_tension(
    alpha1: Any,
    phase: Annotated[dict[str, float], "models"],
    neon_runtime: Annotated[Any, "models"],
) -> Any:
    """Capillary face force ``fSigma = interpolate(sigma*K)*snGrad(alpha1)``.

    Computed from the live ``alpha1`` (interfaceProperties); the curvature is
    fixed across the PISO corrector loop, so consumers fold once per operation.
    """
    return nfb.surface_tension_force(neon_runtime, alpha1, phase["sigma"])


gravityForce = Model("gravityForce")
gravityForce.load(_no_config)


@gravityForce.contributes(interfaceForce)
def buoyancy(rho: Any, ghf: Any) -> Any:
    """Buoyant face force ``-ghf*snGrad(rho)`` (interFoam's gravity term)."""
    return (-1.0 * ghf) * nfb.sn_grad(rho)
