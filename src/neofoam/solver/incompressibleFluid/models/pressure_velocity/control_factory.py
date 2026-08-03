# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Factory helpers that build Python-native pressure-velocity controls.

These factories read the algorithm block of ``system/fvSolution`` through its
typed config (:mod:`neofoam.foam.algorithm_configs`) and return
:class:`PimpleControl` / :class:`SimpleControl` instances from
:mod:`neofoam.algorithms.solution_loop.control`. They replace direct use of
``pybFoam.pimpleControl`` so loop logic stays in Python.

The configs carry the OpenFOAM defaults of every key that used to be a
``getOrDefault`` call site here, and are registered on the owning spec (see
``pimpleAlgorithm`` / ``simpleAlgorithm``) so the whole key set is exported to
``configurations(solver)`` and the MCP instead of only being read at run time.
"""

from typing import Any

import pybFoam as pyf

from neofoam.algorithms.solution_loop.control import PimpleControl, SimpleControl
from neofoam.foam.algorithm_configs import (
    DynamicMeshControls,
    PimpleAlgorithmConfig,
    PisoAlgorithmConfig,
    PisoDynamicMeshControls,
    SimpleAlgorithmConfig,
)

#: The two classes describing one algorithm block: its loop controls and its
#: mesh-motion switches. ``IOMetadata.subdict`` is one fixed string per class,
#: so the ``PISO`` spelling of the block needs its own pair.
_PIMPLE_CLASSES = (PimpleAlgorithmConfig, DynamicMeshControls)
_PISO_CLASSES = (PisoAlgorithmConfig, PisoDynamicMeshControls)


def _algorithm_classes() -> tuple[type[PimpleAlgorithmConfig], type[DynamicMeshControls]]:
    """The config classes for whichever of ``PIMPLE`` / ``PISO`` the case ships.

    pisoFoam tutorials ship a ``PISO`` block where pimpleFoam ships ``PIMPLE``;
    PISO is a single-outer-loop PIMPLE, so it is read from the same place.
    """
    fv_solution = pyf.dictionary.read("system/fvSolution")
    if fv_solution.isDict("PIMPLE"):
        return _PIMPLE_CLASSES
    if fv_solution.isDict("PISO"):
        return _PISO_CLASSES
    raise ValueError(
        "incompressibleFluid: system/fvSolution has neither a PIMPLE nor a "
        "PISO block to build the pressure-velocity control from."
    )


def load_pimple_config() -> PimpleAlgorithmConfig:
    """The active algorithm block as a typed config.

    A ``PISO`` block pins ``nOuterCorrectors`` to 1 instead of reading it — PISO
    has no outer loop. ``validate=False`` mirrors the framework's own auto-load:
    absent keys fall back to the schema defaults, present ones are coerced to
    the declared types.
    """
    config_cls, _ = _algorithm_classes()
    config = config_cls.load(validate=False)
    if issubclass(config_cls, PisoAlgorithmConfig):
        return config.model_copy(update={"nOuterCorrectors": 1})
    return config


def create_dynamic_mesh_controls(context: dict[str, Any]) -> DynamicMeshControls:
    """The algorithm block's mesh-motion switches (``createDyMControls.H``).

    Native re-reads them every time step (``readDyMControls.H``); read once here
    because no tutorial rewrites them mid-run. ``correctPhi``'s OpenFOAM default
    is ``mesh.dynamic()``, resolved here where the mesh is in hand, so the
    operations that consume the switches see a plain bool.
    """
    _, controls_cls = _algorithm_classes()
    controls = controls_cls.load(validate=False)
    return controls.resolved(mesh_dynamic=bool(context["mesh"].dynamic()))


def create_pimple_control(_context: dict[str, Any]) -> PimpleControl:
    """Create a :class:`PimpleControl` from the PIMPLE (or PISO) block."""
    config = load_pimple_config()
    return PimpleControl(
        nOuterCorrectors=config.nOuterCorrectors,
        nCorrectors=config.nCorrectors,
        nNonOrthogonalCorrectors=config.nNonOrthogonalCorrectors,
        momentumPredictor=config.momentumPredictor,
        turbCorr=config.turbCorr,
        # pimpleControl::read() defaults: turbOnFinalIterOnly true,
        # finalOnLastPimpleIterOnly false.
        turbOnFinalIterOnly=config.turbOnFinalIterOnly,
        finalOnLastPimpleIterOnly=config.finalOnLastPimpleIterOnly,
    )


def create_simple_control(_context: dict[str, Any]) -> SimpleControl:
    """Create a :class:`SimpleControl` from the SIMPLE block."""
    config = SimpleAlgorithmConfig.load(validate=False)
    return SimpleControl(
        nNonOrthogonalCorrectors=config.nNonOrthogonalCorrectors,
        momentumPredictor=config.momentumPredictor,
        consistent=config.consistent,
        useResidualConvergence=False,
    )
