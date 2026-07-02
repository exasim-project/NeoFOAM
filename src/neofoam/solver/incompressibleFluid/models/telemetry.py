# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""telemetry — opt-in OpenTelemetry performance tracing for a case.

A pure-config optional model: it owns the ``system/controlDict``
``telemetry`` sub-dict and its ``@detect`` mirrors OpenFOAM conventions
(the dict present and ``enabled`` absent-or-true activates it). It has no
build step and no operations — activation happens in the solver's ``run()``
via :func:`maybe_configure_telemetry` *before* initialization, so init
spans are captured too. The framework's operation/init instrumentation and
the per-rank span files are provided by :mod:`neofoam.telemetry`.
"""

import os
from pathlib import Path
from typing import Union

from pybFoam import dictionary

from neofoam import telemetry as telemetry_shim
from neofoam.io import OF, BaseConfig, IOStrategy
from neofoam.telemetry import TelemetrySettings

from .incompressibleFluidModel import Model, incompressibleFluidModel


@IOStrategy(OF("system/controlDict", subdict="telemetry"))
class TelemetryDictConfig(BaseConfig):
    """The ``telemetry`` sub-dict of ``system/controlDict`` (all keys optional)."""

    enabled: bool = True
    directory: str = "telemetry"
    summary: bool = True


telemetry = Model("telemetry").register_with(incompressibleFluidModel)
telemetry.config(TelemetryDictConfig)


@telemetry.detect
def detect_model() -> bool:
    """Active iff ``system/controlDict`` has a ``telemetry`` dict that enables it.

    Read relative to the run's working directory; inactive when the file or
    the dict is absent, or when the dict says ``enabled no``.
    """
    if not os.path.isfile("system/controlDict"):
        return False
    cd = dictionary.read("system/controlDict")
    if not cd.found("telemetry"):
        return False
    sub = cd.subDict("telemetry")
    return not sub.found("enabled") or sub.get[bool]("enabled")


def maybe_configure_telemetry(case_dir: Union[Path, str] = ".") -> bool:
    """Activate the telemetry shim when the case opts in; returns whether it did.

    Raises :class:`neofoam.telemetry.TelemetryNotInstalledError` when the case
    enables telemetry but the optional ``neofoam[telemetry]`` extra is missing.
    """
    cwd = os.getcwd()
    os.chdir(case_dir)
    try:
        if not telemetry.run_detect():
            return False
    finally:
        os.chdir(cwd)

    config = TelemetryDictConfig.load(case_dir=case_dir)
    telemetry_shim.configure(
        TelemetrySettings(
            enabled=config.enabled,
            directory=config.directory,
            summary=config.summary,
            service_name="incompressibleFluid",
        ),
        case_dir=case_dir,
    )
    return True
