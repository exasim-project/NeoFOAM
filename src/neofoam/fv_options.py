# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Finite-volume options (``fvOptions``) momentum sources, as an optional model.

Use it when a case's physics *is* a source term declared in an ``fvOptions``
dictionary — a rotor disk, a porosity block, a mean-velocity force. The model is
*detected*: without that file it is never instantiated and the pressure-velocity
algorithms, which inject it optionally and branch on ``None``, assemble exactly the
equations they did before this model existed.

It owns one runtime object, ``ctx.models["fv_options"]`` — OpenFOAM's own
:class:`Foam::fv::options` — whose hooks (the source matrix, ``constrain``,
``correct``) the algorithms apply where native's ``UEqn.H``/``pEqn.H`` apply them.
The incompressibleFluid algorithms reach them through the extension points their
operations declare — this spec registers the implementations there;
incompressibleVoF still injects the model optionally and branches on ``None``.
Shared by ``incompressibleFluid`` and ``incompressibleVoF``, which each register
this one spec with their own plugin family.

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

# OpenFOAM's own search order (``fv::options::createIOobject``). Case-relative:
# the solver runs with the case directory as its working directory.
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

    Declared so the file is part of the solver's config schema. The entries stay
    free-form: their values are consumed by ``fv::options`` straight off disk, so
    modelling the dozens of ``fv::option`` arms in pydantic would buy nothing. The
    ``system/`` location is handled by the spec's ``@load`` below.

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

    Via ``fv::options::New``, which caches on the mesh registry — the native
    turbulence closures source their own equations from that same object, and a
    second list would double-apply every source. Named ``fv_options`` because the
    model runtime itself already occupies ``models.fvOptions``.
    """

    def create_fv_options(context: dict[str, Any]) -> pyf.fvOptions:
        return pyf.fvOptions.New(context["mesh"])

    return [model("fv_options", create_fv_options, depends_on=["mesh"])]
