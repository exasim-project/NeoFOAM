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

from neofoam.io import collect_config_classes


def config_classes() -> list[type]:
    """All config classes the incompressibleFluid solver may consume.

    Static — needs no case directory. Optional-model classes are listed for
    every registered model (detection, which needs a case, is not run).
    """
    from .configs import ControlDictConfig, TransportPropertiesConfig
    from .models.incompressibleFluidModel import incompressibleFluidModel
    from .models.pressure_velocity.pimpleAlgorithm import pimple

    sources: list[object] = [
        ControlDictConfig,
        TransportPropertiesConfig,
        pimple,
        *incompressibleFluidModel.all_specs(),
    ]
    return collect_config_classes(sources)
