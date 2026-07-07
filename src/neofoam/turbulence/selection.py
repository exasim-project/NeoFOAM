# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turbulence model selection / factory.

Resolves the active turbulence model name from a turbulenceProperties-like
config and returns either a registered native NeoFOAM model spec or an OpenFOAM
fallback adapter.

:func:`select_turbulence_model` operates on a *duck-typed* config (any object
exposing ``simulationType`` and optional ``RAS`` / ``LES`` sub-objects), so this
module stays free of ``neofoam.io`` / pybFoam and is unit-testable without
OpenFOAM. :func:`select_from_case`, which loads the real OpenFOAM dictionary, is
the only pybFoam-bound entry point.
"""

from pathlib import Path
from typing import Any, Optional, Union

from .fallback import OpenFOAMTurbulenceModel, TurbulenceFactory
from .momentumTransport import ModelSpec, momentumTransportModel

__all__ = ["model_name", "select_turbulence_model", "select_from_case"]

SelectedModel = Union[ModelSpec, OpenFOAMTurbulenceModel]


def model_name(config: Any) -> Optional[str]:
    """Resolve the active model name from a turbulenceProperties config.

    Returns ``"laminar"`` for ``simulationType laminar``, the ``RASModel`` for
    ``RAS``, the ``LESModel`` for ``LES``, or ``None`` when it cannot be
    determined (unknown ``simulationType`` or a missing sub-dictionary).
    """
    sim_type = getattr(config, "simulationType", None)
    if sim_type == "laminar":
        return "laminar"
    if sim_type == "RAS":
        ras = getattr(config, "RAS", None)
        return getattr(ras, "RASModel", None) if ras is not None else None
    if sim_type == "LES":
        les = getattr(config, "LES", None)
        return getattr(les, "LESModel", None) if les is not None else None
    return None


def select_turbulence_model(
    config: Any,
    *,
    U: Any = None,
    phi: Any = None,
    transport: Any = None,
    of_factory: Optional[TurbulenceFactory] = None,
) -> SelectedModel:
    """Select a turbulence model from a turbulenceProperties config.

    Returns the registered native :class:`ModelSpec` whose name matches the
    configured model, or an :class:`OpenFOAMTurbulenceModel` fallback when no
    native model is registered for that name.
    """
    name = model_name(config)
    spec = momentumTransportModel.find_spec(name) if name is not None else None
    if spec is not None:
        return spec
    return OpenFOAMTurbulenceModel(U, phi, transport, factory=of_factory)


def select_from_case(
    case_dir: Union[str, Path] = ".",
    *,
    U: Any = None,
    phi: Any = None,
    transport: Any = None,
    of_factory: Optional[TurbulenceFactory] = None,
) -> SelectedModel:
    """Load ``constant/turbulenceProperties`` from a case and select the model.

    This is the pybFoam-bound entry point: it imports and uses the OpenFOAM
    reading strategy via :class:`TurbulencePropertiesConfig`.
    """
    from .config import TurbulencePropertiesConfig

    config = TurbulencePropertiesConfig.load(case_dir=case_dir)
    return select_turbulence_model(
        config, U=U, phi=phi, transport=transport, of_factory=of_factory
    )
