# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Viscosity (transport) model selection / factory.

Resolves the active transport model name from a transportProperties-like config
and returns either a registered native NeoFOAM viscosity model spec or an
OpenFOAM fallback adapter.

:func:`select_viscosity_model` operates on a *duck-typed* config (any object
exposing ``transportModel``), so this module stays free of ``neofoam.io`` /
pybFoam and is unit-testable without OpenFOAM. :func:`select_from_case`, which
loads the real OpenFOAM dictionary, is the only pybFoam-bound entry point.
"""

from pathlib import Path
from typing import Any, Optional, Union

from .fallback import OpenFOAMViscosityModel, TransportFactory
from .viscosityModel import ModelSpec, viscosityModel

__all__ = ["model_name", "select_viscosity_model", "select_from_case"]

SelectedModel = Union[ModelSpec, OpenFOAMViscosityModel]


def model_name(config: Any) -> Optional[str]:
    """Resolve the transport model name from a transportProperties config.

    Returns the ``transportModel`` entry (e.g. ``"Newtonian"``,
    ``"CrossPowerLaw"``), or ``None`` when it cannot be determined.
    """
    return getattr(config, "transportModel", None)


def select_viscosity_model(
    config: Any,
    *,
    U: Any = None,
    phi: Any = None,
    of_factory: Optional[TransportFactory] = None,
) -> SelectedModel:
    """Select a viscosity model from a transportProperties config.

    Returns the registered native :class:`ModelSpec` whose name matches the
    configured ``transportModel``, or an :class:`OpenFOAMViscosityModel`
    fallback when no native model is registered for that name.
    """
    name = model_name(config)
    spec = viscosityModel.find_spec(name) if name is not None else None
    if spec is not None:
        return spec
    return OpenFOAMViscosityModel(U, phi, factory=of_factory)


def select_from_case(
    case_dir: Union[str, Path] = ".",
    *,
    U: Any = None,
    phi: Any = None,
    of_factory: Optional[TransportFactory] = None,
) -> SelectedModel:
    """Load ``constant/transportProperties`` from a case and select the model.

    This is the pybFoam-bound entry point: it imports and uses the OpenFOAM
    reading strategy via :class:`TransportPropertiesConfig`.
    """
    from .config import TransportPropertiesConfig

    config = TransportPropertiesConfig.load(case_dir=case_dir)
    return select_viscosity_model(config, U=U, phi=phi, of_factory=of_factory)
