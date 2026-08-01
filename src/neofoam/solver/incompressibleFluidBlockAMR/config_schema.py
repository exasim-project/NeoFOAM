# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-free config schema for the incompressibleFluidBlockAMR solver.

:func:`config_classes` returns every ``BaseConfig`` class the solver may
consume — solver-core configs plus the configs of every model family it owns —
**without a case directory**.
"""

from __future__ import annotations


def config_classes() -> list[type]:
    """All config classes the incompressibleFluidBlockAMR solver may consume.

    Static — needs no case directory. Thin wrapper over the framework's
    solver-agnostic :func:`neofoam.framework.solver.configurations`.
    """
    from neofoam.framework.solver import configurations  # noqa: PLC0415  # cycle: framework

    from .incompressibleFluidBlockAMR import (  # noqa: PLC0415  # circular: solver module
        incompressibleFluidBlockAMR,
    )

    return list(configurations(incompressibleFluidBlockAMR))
