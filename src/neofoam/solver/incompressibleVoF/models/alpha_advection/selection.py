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

No upstream interIsoFoam tutorial declares ``advectionScheme`` (it is a
NeoFOAM-only key), so when it is absent ``select_from_case`` falls back to
inspecting the ``solvers`` "alpha.*" sub-dict for isoAdvector-only controls
(``reconstructionScheme``/``isoFaceTol``/``surfCellTol``/``nAlphaBounds``) —
an exact MULES/isoAdvector discriminator over the upstream tutorial corpus.
"""

from typing import Any

import pybFoam as pyf
from pybFoam import Info

from .advectionModel import ModelSpec, advectionModel

__all__ = ["model_name", "select_advection_scheme", "select_from_case"]

_DEFAULT_SCHEME = "MULES"
_ISO_ADVECTOR_SCHEME = "isoAdvector"
# Keys that only ever appear in an isoAdvector "alpha.*" solver sub-dict
# (Foam::isoAdvection's construction-time controls); MULES cases never carry
# any of these.
_ISO_ADVECTOR_DISCRIMINATOR_KEYS = (
    "reconstructionScheme",
    "isoFaceTol",
    "surfCellTol",
    "nAlphaBounds",
)


def model_name(config: Any) -> str:
    """Resolve the advection-scheme name from a duck-typed config.

    Returns the ``advectionScheme`` attribute, or ``"MULES"`` when absent.
    """
    return str(getattr(config, "advectionScheme", _DEFAULT_SCHEME))


def select_advection_scheme(name: str) -> ModelSpec:
    """Return the registered spec for ``name``.

    Raises ``ValueError`` when ``name`` is not a registered scheme — a silent
    fallback to MULES would disguise a routing failure as a scheme error.
    """
    spec = advectionModel.find_spec(name)
    if spec is None:
        raise ValueError(
            f"Unknown advectionScheme '{name}'; registered: {advectionModel.registered_names()}"
        )
    return spec


def _isoadvector_controls_present(fv_solution: Any) -> bool:
    """True when ``solvers`` has an "alpha.*" sub-dict with isoAdvector controls."""
    if not fv_solution.found("solvers"):
        return False
    solvers = fv_solution.subDict("solvers")
    for key in solvers.toc():
        name = str(key)
        if not name.startswith("alpha.") or not solvers.isDict(name):
            continue
        alpha_dict = solvers.subDict(name)
        if any(alpha_dict.found(k) for k in _ISO_ADVECTOR_DISCRIMINATOR_KEYS):
            return True
    return False


def select_from_case(case_dir: str = ".") -> ModelSpec:
    """Read ``advectionScheme`` from ``system/fvSolution`` and select the scheme.

    An explicit ``advectionScheme`` key always wins. Otherwise isoAdvector is
    selected when the ``solvers`` "alpha.*" sub-dict carries isoAdvector
    controls (see :data:`_ISO_ADVECTOR_DISCRIMINATOR_KEYS`); MULES otherwise.
    A missing/unreadable fvSolution also falls back to MULES, logged so it
    never silently changes the scheme.
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
        if fv_solution.found("advectionScheme"):
            name = str(fv_solution.getOrDefault[str]("advectionScheme", _DEFAULT_SCHEME))
        elif _isoadvector_controls_present(fv_solution):
            name = _ISO_ADVECTOR_SCHEME
    return select_advection_scheme(name)
