# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Every config an in-tree solver reads is reachable through the framework itself.

A ``BaseConfig`` only becomes part of a solver's authoring surface when its
owning spec *declares* it (``spec.config(Cls)``). That declaration is the single
link in the chain ``ModelSpec._config_classes`` → ``collect_config_classes`` →
``configurations(solver)`` → :func:`neofoam.io.schema.list_configs` (the MCP's
``list_configs`` / ``config_schema`` and the ``save_case`` envelope). A class
that is merely constructed inside a ``@load`` or a ``@build`` still works at run
time but is invisible to every one of those consumers — the failure mode this
module exists to catch.

So the assertion is deliberately a *superset* one on the discovery path itself:
adding a config is free, dropping (or forgetting) a registration fails a named
test. The listed names are the ones each solver's models actually read; per-file
detail lives with each class's own spec (``test/foam/test_algorithm_configs.py``,
``test/turbulence/…``), and the "every required case file has an owner" angle is
covered by ``test/tooling/workflow/test_configurations_complete.py``.

Hermetic and case-free: ``configurations`` never touches a case directory.
"""

from typing import Any

import pytest

from neofoam.io.schema import list_configs
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid
from neofoam.solver.incompressibleFluidNeoN.incompressibleFluidNeoN import (
    incompressibleFluidNeoN,
)
from neofoam.solver.incompressibleVoF.incompressibleVoF import incompressibleVoF

pytest.importorskip("pybFoam")  # resolving a solver spec imports pybFoam

#: Config classes each solver's models read at run time and must therefore
#: declare. Pre-existing entries are listed alongside the newer typed ones so a
#: registration cannot silently disappear from either group.
_EXPECTED: dict[str, set[str]] = {
    "incompressibleFluid": {
        # solver core
        "ControlDictConfig",
        "TelemetryDictConfig",
        "PreprocessConfig",
        "BlockMeshDictConfig",
        "SnappyHexMeshDictConfig",
        # pressure-velocity: the per-spec fvSchemes/fvSolution slices …
        "Pimple_fvSchemes",
        "Pimple_fvSolution",
        "Simple_fvSchemes",
        "Simple_fvSolution",
        # … and the algorithm blocks those slices pass through untyped
        "PimpleAlgorithmConfig",
        "DynamicMeshControls",
        "PisoAlgorithmConfig",
        "PisoDynamicMeshControls",
        "SimpleAlgorithmConfig",
        # transport + turbulence, closure coefficients included
        "TransportPropertiesConfig",
        "TurbulencePropertiesConfig",
        "KEpsilonCoeffs",
        "KOmegaSSTCoeffs",
        "SpalartAllmarasCoeffs",
        # optional models
        "BoussinesqConfig",
        "GravityConfig",
        "CourantConfig",
        "MaxDeltaTConfig",
        "MRFPropertiesConfig",
        "FvOptionsConfig",
        # 0/<field> schemas synthesised from the field declarations
        "UFieldConfig",
        "pFieldConfig",
    },
    "incompressibleVoF": {
        "ControlDictConfig",
        "TransportPropertiesConfig",
        "GravityConfig",
        "TurbulencePropertiesConfig",
        "MULES_fvSchemes",
        "MULES_fvSolution",
        "Pimple_fvSchemes",
        "Pimple_fvSolution",
        "VofPimpleAlgorithmConfig",
        "DynamicMeshControls",
        "PorosityPropertiesConfig",
        "MRFPropertiesConfig",
        "FvOptionsConfig",
        "UFieldConfig",
        "p_rghFieldConfig",
        "alpha.waterFieldConfig",
    },
    "incompressibleFluidNeoN": {
        "ControlDictConfig",
        "NeoNControlConfig",
        "TransportPropertiesConfig",
        "TurbulencePropertiesConfig",
        "KEpsilonCoeffs",
        "KOmegaSSTCoeffs",
        "SpalartAllmarasCoeffs",
        "PimpleNeoN_fvSchemes",
        "PimpleNeoN_fvSolution",
        "SimpleNeoN_fvSchemes",
        "SimpleNeoN_fvSolution",
        "UFieldConfig",
        "pFieldConfig",
    },
}

_SOLVERS: dict[str, Any] = {
    "incompressibleFluid": incompressibleFluid,
    "incompressibleVoF": incompressibleVoF,
    "incompressibleFluidNeoN": incompressibleFluidNeoN,
}


@pytest.mark.parametrize("solver_name", sorted(_EXPECTED))
def test_every_expected_config_is_declared_on_the_solver(solver_name: str) -> None:
    declared = {info.cls_name for info in list_configs(_SOLVERS[solver_name])}
    missing = _EXPECTED[solver_name] - declared
    assert not missing, (
        f"{solver_name}: not reachable through configurations()/list_configs — "
        f"register them with spec.config(...): {sorted(missing)}"
    )
