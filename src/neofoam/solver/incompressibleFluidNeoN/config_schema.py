# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-free config schema for the incompressibleFluidNeoN solver.

:func:`config_classes` returns every ``BaseConfig`` class the solver may
consume — solver-core configs, the PIMPLE ``fvSchemes`` / ``fvSolution``
slices, and the configs of every registered optional model — **without a
case directory**.
"""

from __future__ import annotations


def config_classes() -> list[type]:
    """All config classes the incompressibleFluidNeoN solver may consume.

    Static — needs no case directory. Thin wrapper over the framework's
    solver-agnostic :func:`neofoam.framework.solver.configurations`.
    """
    from neofoam.framework.solver import configurations

    from .incompressibleFluidNeoN import incompressibleFluidNeoN

    return list(configurations(incompressibleFluidNeoN))
