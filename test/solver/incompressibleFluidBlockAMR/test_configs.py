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

from pydantic import ValidationError  # noqa: E402

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
        "backend": "jax",
        "verbose": 0,
        "bottomVerbose": 0,
    }
    # empty ibm / bottomSolver are dropped from the resolved dict
    assert "ibm" not in sol_p
    assert "bottomSolver" not in sol_p


def test_solvers_u_resolve_cylinder_keeps_ibm():
    sol_U = USolutionConfig.load(case_dir=CASES / "cylinder").resolve()
    assert sol_U["ibm"] == "directForcing"
    assert sol_U["backend"] == "jax"


def test_solvers_u_resolve_box_has_no_ibm():
    sol_U = USolutionConfig.load(case_dir=CASES / "box").resolve()
    assert "ibm" not in sol_U


def test_solvers_p_resolve_cylinder_keeps_bottomsolver():
    sol_p = PSolutionConfig.load(case_dir=CASES / "cylinder").resolve()
    assert sol_p["bottomSolver"] == "bicgstab"


def test_backend_rejects_unknown_value():
    with pytest.raises(ValidationError):
        USolutionConfig(backend="foo")


def test_config_schema_lists_new_configs():
    classes = config_classes()
    assert FvSchemesConfig in classes
    assert USolutionConfig in classes
    assert PSolutionConfig in classes
