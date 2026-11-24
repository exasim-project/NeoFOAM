# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Config round-trips for the fvSchemes + per-field fvSolution configs (plan 05).

Loads the real bundled case files under ``cases/`` — the OpenFOAM reader binds
``system/fvSchemes`` and the nested ``solvers.U`` / ``solvers.p`` subdicts of
``system/fvSolution`` — and checks the flattened DSL dicts that reach the engine.
"""

from pathlib import Path

import pytest

pytest.importorskip("neon")

from neofoam.solver.incompressibleFluidBlockAMR.config_schema import (  # noqa: E402
    config_classes,
)
from neofoam.solver.incompressibleFluidBlockAMR.configs import (  # noqa: E402
    FvSchemesConfig,
    PSolutionConfig,
    USolutionConfig,
)

CASES = Path(__file__).parent / "cases"


def test_fvschemes_resolve_flattens_with_default_expansion():
    schemes = FvSchemesConfig.load(case_dir=CASES / "box").resolve()
    # ``default`` entries expand to the bare operator key; specific keys verbatim.
    assert schemes == {
        "ddt": "Euler",
        "div(phi,U)": "vanLeer",
        "laplacian": "central",
        "grad": "central",
    }


def test_solvers_p_resolve_box():
    sol_p = PSolutionConfig.load(case_dir=CASES / "box").resolve()
    assert sol_p == {
        "solver": "MLMG",
        "rtol": 1e-10,
        "atol": 1e-8,
        "maxIter": 200,
        "verbose": 0,
    }


def test_config_schema_lists_new_configs():
    classes = config_classes()
    assert FvSchemesConfig in classes
    assert USolutionConfig in classes
    assert PSolutionConfig in classes
