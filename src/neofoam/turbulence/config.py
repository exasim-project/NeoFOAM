# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``constant/turbulenceProperties`` configuration.

Reads the OpenFOAM turbulence dictionary into a validated pydantic model via the
OpenFOAM IO strategy. ``simulationType`` selects RAS / LES / laminar; the
``RAS`` / ``LES`` sub-dictionaries map to nested ``BaseModel`` sub-configs —
:class:`~neofoam.io.strategies.openfoam_strategy.OpenFOAMStrategy` recurses into
sub-dictionaries automatically when a field's type is a ``BaseModel`` subclass.

This is the only turbulence module that imports ``neofoam.io`` (hence pybFoam);
tests that load it run the OpenFOAM IO path directly.
"""

from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from neofoam.io import OF, BaseConfig, IOStrategy

__all__ = [
    "RASProperties",
    "LESProperties",
    "TurbulencePropertiesConfig",
    "model_coefficients",
]


class RASProperties(BaseModel):
    """The ``RAS`` sub-dictionary of ``turbulenceProperties``.

    ``extra="allow"`` keeps the per-model ``<RASModel>Coeffs`` sub-dictionary
    OpenFOAM selects as ``RASModel::coeffDict_``; read it with
    :func:`model_coefficients`.
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


def model_coefficients(config: Any, model: str, defaults: Mapping[str, float]) -> dict[str, float]:
    """A closure's coefficients, with the case's ``<model>Coeffs`` overrides applied.

    Mirrors OpenFOAM's ``RASModel::coeffDict_`` (``subOrEmptyDict(type + "Coeffs")``)
    read through ``dimensioned::getOrAddToDict``: a coefficient the sub-dictionary
    sets wins, one it omits keeps the closure default, and an entry the closure does
    not declare is ignored — OpenFOAM never rejects an unknown coefficient either.

    Use it from a closure's ``@build`` so the per-step operations read one resolved
    mapping rather than module constants. The ``RAS`` block arrives as a
    :class:`RASProperties` from :meth:`~neofoam.io.BaseConfig.load` but as a plain
    ``dict`` from ``ModelSpec.instantiate``, so both shapes are accepted.

    Example:
        ``model_coefficients(config, "kEpsilon", {"Cmu": 0.09, "sigmaEps": 1.3})``
    """
    ras = getattr(config, "RAS", None)
    key = f"{model}Coeffs"
    overrides = ras.get(key) if isinstance(ras, Mapping) else getattr(ras, key, None)
    if not isinstance(overrides, Mapping):
        return dict(defaults)
    return {
        name: float(overrides[name]) if name in overrides else default
        for name, default in defaults.items()
    }
