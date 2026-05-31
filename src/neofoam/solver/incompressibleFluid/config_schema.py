# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-free config schema for the incompressibleFluid solver.

:func:`config_classes` returns every ``BaseConfig`` class the solver may
consume — solver-core configs, the PIMPLE ``fvSchemes`` / ``fvSolution``
slices, and the configs of every registered optional model — **without a
case directory**. Fill these classes and write them with
:func:`neofoam.io.save_configs` to scaffold a case from scratch (e.g. by an
agent). The loaded *instances* for a concrete case come from
``create_init(case_dir).run_load().configs`` after LOAD/RESOLVE instead.
"""

from __future__ import annotations


def config_classes() -> list[type]:
    """All config classes the incompressibleFluid solver may consume.

    Static — needs no case directory. Thin wrapper over the framework's
    solver-agnostic :func:`neofoam.configurations`: the schema is derived
    from the configs and model families declared on the ``incompressibleFluid``
    spec at import. Optional-model classes are listed for every registered
    model (detection, which needs a case, is not run).
    """
    from neofoam.framework.solver import configurations

    from .incompressibleFluid import incompressibleFluid

    return list(configurations(incompressibleFluid))
