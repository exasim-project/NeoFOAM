# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Finite-volume options (``fvOptions``) momentum sources, as an optional model.

Use it when a case's physics *is* a source term declared in an ``fvOptions``
dictionary — a rotor disk, a porosity block, an actuation disk, a mean-velocity
force, mangrove drag. The model is *detected*: without that file it is never
instantiated, nothing lands on the Context, and every pressure-velocity algorithm
assembles exactly the equations it assembled before this model existed.

The model owns one runtime object, ``ctx.models["fv_options"]`` — OpenFOAM's own
:class:`Foam::fv::options`, which reads the dictionary, selects the source types
and implements the three hooks native's ``UEqn.H``/``pEqn.H`` use: the source
matrix ``fvOptions(U)`` (``fvOptions(rho, U)`` for VoF), ``constrain(UEqn)``
*after* the equation is relaxed, and ``correct(U)`` *after* each solve of U. The
algorithms that consume it (SIMPLE and both PIMPLEs) take it as an *optional*
injected model and branch on ``None``, so this module never has to supply a no-op
stand-in.

Shared by ``incompressibleFluid`` and ``incompressibleVoF``: each solver's model
package registers this one spec with its own plugin family (the source call
differs — ``fvOptions(U)`` for the single-phase solvers, ``fvOptions(rho, U)``
for VoF — but that lives at the call site, not here).

Example::

    from neofoam.fv_options import fvOptions
    fvOptions.register_with(incompressibleFluidModel)
"""

from pathlib import Path
from typing import Any, Optional

import pybFoam as pyf
from pydantic import ConfigDict

from neofoam.framework.initialization import model
from neofoam.framework.model import Model
from neofoam.io import OF, BaseConfig, IOStrategy

__all__ = ["FvOptionsConfig", "fvOptions"]

# OpenFOAM's own search order (``fv::options::createIOobject``): ``constant/``
# first, ``system/`` as the fallback. The dictionary is read by OpenFOAM, never
# by Python, so detection is a file-existence question and the paths are
# case-relative — the solver runs with the case directory as its working
# directory.
_FV_OPTIONS_PATHS = ("constant/fvOptions", "system/fvOptions")


def _dictionary_path() -> Optional[Path]:
    """First of the two locations that exists, in OpenFOAM's search order."""
    for candidate in _FV_OPTIONS_PATHS:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


@IOStrategy(OF(_FV_OPTIONS_PATHS[0]))
class FvOptionsConfig(BaseConfig):
    """``constant/fvOptions`` (or ``system/fvOptions``) — one sub-dict per source.

    Declared so the file is part of the solver's config schema (collectible,
    savable, printable like any other case file). The entries stay free-form: a
    source name maps to a sub-dict whose ``type`` selects one of the dozens of
    entries in OpenFOAM's ``fv::option`` runtime-selection table, each with its
    own coefficients. Modelling those arms in pydantic would buy nothing — the
    values are consumed by ``fv::options`` straight off disk.

    The registered path is the ``constant/`` one because that is where
    ``fv::options`` looks first; a case that keeps the file in ``system/``
    instead is still loaded, by the spec's ``@load`` below.

    Example::

        FvOptionsConfig.load(case_dir=case).model_extra["porosity1"]["type"]
    """

    model_config = ConfigDict(extra="allow")


fvOptions = Model("fvOptions").labeled("Finite volume options (fvOptions)")
fvOptions.config(FvOptionsConfig)


@fvOptions.detect
def detect_model() -> bool:
    """fvOptions is active exactly when the case carries one of the two dictionaries."""
    return _dictionary_path() is not None


@fvOptions.load
def load_config(case_dir: Path, _instance_id: Optional[str]) -> FvOptionsConfig:
    """Load the dictionary from whichever of the two locations the case uses."""
    path = _dictionary_path()
    if path is None:
        raise FileNotFoundError(
            f"fvOptions model instantiated for a case without any of {_FV_OPTIONS_PATHS}"
        )
    return FvOptionsConfig.load(case_dir=case_dir, file=str(path), validate=False)


@fvOptions.build
def build(_config: FvOptionsConfig) -> list[Any]:
    """Publish the option list on the Context as ``models.fv_options``.

    One step, depending only on the mesh. ``fv::options::New`` looks the list up
    on the mesh registry and constructs it there on first call, so this is the
    same object the native turbulence closures source their own ``k``/``epsilon``
    equations from — constructing a second one would double-apply every source.
    Every equation-level hook (the source matrix, ``constrain``, ``correct``) is
    applied by the pressure-velocity algorithm at the point native applies it,
    not here. The name is ``fv_options`` rather than ``fvOptions`` because the
    model runtime itself already occupies ``models.fvOptions``.
    """

    def create_fv_options(context: dict[str, Any]) -> pyf.fvOptions:
        return pyf.fvOptions.New(context["mesh"])

    return [model("fv_options", create_fv_options, depends_on=["mesh"])]
