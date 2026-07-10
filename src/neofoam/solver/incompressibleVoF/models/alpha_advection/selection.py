# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Alpha-advection scheme selection / factory.

Resolves which registered advection scheme is active for a case from the
explicit ``advectionScheme`` key in ``system/fvSolution`` (default ``MULES``),
returning the matching :class:`ModelSpec`.

The discriminator is read straight from ``system/fvSolution`` via
``pyf.dictionary.read`` — mirroring :class:`PressureVelocityAlgorithm` and the
alpha-controls reader — rather than routed through an ``@IOStrategy`` config,
because the key lives nested alongside ``PIMPLE``/``solvers`` and only a single
string is needed to pick the scheme.
"""

from typing import Any

import pybFoam as pyf
from pybFoam import Info

from .advectionModel import ModelSpec, advectionModel

__all__ = ["model_name", "select_advection_scheme", "select_from_case"]

_DEFAULT_SCHEME = "MULES"


def model_name(config: Any) -> str:
    """Resolve the advection-scheme name from a duck-typed config.

    Returns the ``advectionScheme`` attribute, or ``"MULES"`` when absent.
    """
    return str(getattr(config, "advectionScheme", _DEFAULT_SCHEME))


def select_advection_scheme(name: str) -> ModelSpec:
    """Return the registered spec for ``name``, or the MULES fallback.

    Warns (and falls back to MULES) when ``name`` is not a registered scheme.
    """
    spec = advectionModel.find_spec(name)
    if spec is not None:
        return spec

    Info(
        f"Unknown advectionScheme '{name}'; falling back to '{_DEFAULT_SCHEME}'. "
        f"Registered: {advectionModel.registered_names()}"
    )
    fallback = advectionModel.find_spec(_DEFAULT_SCHEME)
    if fallback is None:
        raise ValueError(
            f"No advection scheme registered under '{_DEFAULT_SCHEME}'; "
            f"registered: {advectionModel.registered_names()}"
        )
    return fallback


def select_from_case(case_dir: str = ".") -> ModelSpec:
    """Read ``advectionScheme`` from ``system/fvSolution`` and select the scheme.

    Defaults to ``MULES`` when the key (or the file) is absent; the fallback is
    logged so a missing/unreadable fvSolution never silently changes the scheme.
    """
    name = _DEFAULT_SCHEME
    try:
        fv_solution = pyf.dictionary.read("system/fvSolution")
    except RuntimeError as err:
        # pybFoam raises RuntimeError when the file cannot be opened; a
        # malformed value inside the dict is a fatal OpenFOAM IO error instead.
        Info(
            f"select_from_case: cannot read system/fvSolution ({err}); "
            f"using advectionScheme '{_DEFAULT_SCHEME}'."
        )
    else:
        name = str(fv_solution.getOrDefault[str]("advectionScheme", _DEFAULT_SCHEME))
    return select_advection_scheme(name)
