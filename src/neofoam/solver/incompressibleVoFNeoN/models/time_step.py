# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

# NB: no ``from __future__ import annotations`` — the contributions' ``cfg`` parameter is
# matched by its live annotation.

"""interFoam's adaptive time step on NeoN, as limits on the solution loop."""

from typing import Any

from neofoam import neofoam_bindings as nfb
from neofoam.algorithms.solution_loop.interfaces import (
    VGREAT,
    initialTimeStepConstraint,
    maxTimeStep,
    timeStepConstraint,
)
from neofoam.framework.model import Model
from neofoam.solver.incompressibleVoF.configs import ControlDictConfig

__all__ = ["adaptiveTimeStepNeoN"]

SMALL = 1e-15

# Unlike interFoam, the limits are read from controlDict once, not re-read every step.
adaptiveTimeStepNeoN = Model("adaptiveTimeStepNeoN").labeled("Adaptive time step (Courant)")
adaptiveTimeStepNeoN.config(ControlDictConfig)


def _courant_numbers(alpha1: Any, phi: Any, delta_t: float) -> tuple[float, float]:
    """CourantNo.H's CoNum and alphaCourantNo.H's alphaCoNum, one reduction on the executor."""
    co_num, alpha_co_num = nfb.courant_numbers(alpha1, phi, float(delta_t))
    return float(co_num), float(alpha_co_num)


@adaptiveTimeStepNeoN.contributes(timeStepConstraint)
def _courant_limit(alpha1: Any, phi: Any, deltaT: float, cfg: ControlDictConfig) -> float:
    """setDeltaT.H's undamped target step from the flow and interface Courant numbers."""
    co_num, alpha_co_num = _courant_numbers(alpha1, phi, deltaT)
    print(f"Courant Number max: {co_num!r}", flush=True)
    print(f"Interface Courant Number max: {alpha_co_num!r}", flush=True)
    max_delta_t_factor = min(cfg.maxCo / (co_num + SMALL), cfg.maxAlphaCo / (alpha_co_num + SMALL))
    return max_delta_t_factor * float(deltaT)


@adaptiveTimeStepNeoN.contributes(initialTimeStepConstraint)
def _courant_initial_limit(alpha1: Any, phi: Any, deltaT: float, cfg: ControlDictConfig) -> float:
    """setInitialDeltaT.H: only for a flow already moving at the start (CoNum > SMALL)."""
    co_num, _alpha_co_num = _courant_numbers(alpha1, phi, deltaT)
    print(f"Courant Number max: {co_num!r}", flush=True)
    if co_num <= SMALL:
        return VGREAT
    return cfg.maxCo * float(deltaT) / co_num


@adaptiveTimeStepNeoN.contributes(maxTimeStep)
def _max_delta_t_limit(cfg: ControlDictConfig) -> float:
    """``maxDeltaT``, clipped onto the damped step as setDeltaT.H does."""
    return cfg.maxDeltaT
