# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``constant/turbulenceProperties`` configuration.

Reads the OpenFOAM turbulence dictionary into a validated pydantic model via the
OpenFOAM IO strategy. ``simulationType`` selects RAS / LES / laminar; the
``RAS`` / ``LES`` sub-dictionaries map to nested ``BaseModel`` sub-configs —
:class:`~neofoam.io.strategies.openfoam_strategy.OpenFOAMStrategy` recurses into
sub-dictionaries automatically when a field's type is a ``BaseModel`` subclass.

This module owns the dictionary itself; a closure additionally declares its own
``<Model>Coeffs`` :class:`~neofoam.io.BaseConfig` next to the closure and resolves
it here with :func:`model_coefficients`. Loading either runs the OpenFOAM IO path
(hence pybFoam) directly.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, model_validator

from neofoam.io import OF, BaseConfig, IOStrategy

__all__ = [
    "RASProperties",
    "LESProperties",
    "TurbulencePropertiesConfig",
    "model_coefficients",
    "load_with_coefficients",
]

#: A closure's coefficients config — declared next to the closure it parameterises.
CoeffsT = TypeVar("CoeffsT", bound=BaseConfig)


class RASProperties(BaseModel):
    """The ``RAS`` sub-dictionary of ``turbulenceProperties``.

    ``extra="allow"`` keeps the per-model ``<RASModel>Coeffs`` sub-dictionary; read
    it with :func:`model_coefficients`.
    """

    model_config = ConfigDict(extra="allow")

    RASModel: str
    turbulence: bool = True
    printCoeffs: bool = False


class LESProperties(BaseModel):
    """The ``LES`` sub-dictionary of ``turbulenceProperties``."""

    LESModel: str
    turbulence: bool = True
    delta: str = "cubeRootVol"


@IOStrategy(OF("constant/turbulenceProperties"))
class TurbulencePropertiesConfig(BaseConfig):
    """Top-level ``constant/turbulenceProperties`` dictionary.

    ``simulationType`` is the closed set ``{laminar, RAS, LES}`` so the generated
    JSON Schema advertises the three valid options as an enum — the
    :class:`~neofoam.io.strategies.openfoam_strategy.OpenFOAMStrategy` unwraps
    ``Literal[...]`` to ``str`` for disk I/O, so the on-disk format is unchanged.
    A ``model_validator`` then pins the simulationType ⇒ sub-block invariant the
    dispatcher (``selection.model_name``) silently relies on.
    """

    simulationType: Literal["laminar", "RAS", "LES"]
    RAS: Optional[RASProperties] = None
    LES: Optional[LESProperties] = None

    @model_validator(mode="after")
    def _check_simulation_type_consistency(self) -> "TurbulencePropertiesConfig":
        if self.simulationType == "laminar":
            if self.RAS is not None or self.LES is not None:
                raise ValueError("simulationType=laminar must not set RAS or LES sub-dictionary")
        elif self.simulationType == "RAS":
            if self.RAS is None:
                raise ValueError("simulationType=RAS requires the RAS sub-dictionary")
            if self.LES is not None:
                raise ValueError("simulationType=RAS must not set LES sub-dictionary")
        elif self.simulationType == "LES":
            if self.LES is None:
                raise ValueError("simulationType=LES requires the LES sub-dictionary")
            if self.RAS is not None:
                raise ValueError("simulationType=LES must not set RAS sub-dictionary")
        return self


def model_coefficients(config: Any, model: str, coeffs_type: type[CoeffsT]) -> CoeffsT:
    """A closure's typed coefficients, with the case's ``<model>Coeffs`` overrides applied.

    Every field of *coeffs_type* carries the closure's OpenFOAM default, so an entry
    the case omits keeps it. An entry *coeffs_type* does not declare is ignored, as
    OpenFOAM's ``RASModel::coeffDict_`` ignores it. The ``RAS`` block arrives as a
    :class:`RASProperties` from :meth:`~neofoam.io.BaseConfig.load` but as a plain
    ``dict`` from ``ModelSpec.instantiate``, so both shapes are accepted.

    Example:
        ``model_coefficients(config, "kEpsilon", KEpsilonCoeffs)``
    """
    ras = getattr(config, "RAS", None)
    key = f"{model}Coeffs"
    overrides = ras.get(key) if isinstance(ras, Mapping) else getattr(ras, key, None)
    if not isinstance(overrides, Mapping):
        return coeffs_type()
    declared = coeffs_type.model_fields
    return coeffs_type(**{name: float(v) for name, v in overrides.items() if name in declared})


def load_with_coefficients(
    case_dir: Union[str, Path],
    model: str,
    coeffs_type: type[CoeffsT],
) -> SimpleNamespace:
    """A closure's runtime config: the dictionary plus its resolved coefficients.

    The ``@<model>.load`` body of every parameterised closure. Returning both as a
    ``SimpleNamespace`` is what ``ModelSpec.instantiate`` produces for a multi-config
    spec, so ``config_injection`` finds either **by type**: an ``@operation``
    parameter annotated ``coeffs: KEpsilonCoeffs`` is bound without a magic name.
    ``validate=False`` mirrors the auto-load ``ModelSpec.instantiate`` performs.
    """
    properties = TurbulencePropertiesConfig.load(case_dir=case_dir, validate=False)
    return SimpleNamespace(
        properties=properties,
        coeffs=model_coefficients(properties, model, coeffs_type),
    )
