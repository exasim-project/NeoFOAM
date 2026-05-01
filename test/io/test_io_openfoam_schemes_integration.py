# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Integration tests: fvSchemes loading and mutation via pybFoam dictionary.

These tests verify the full pipeline:
1. Load valid fvSchemes.of via OpenFOAMStrategy
2. Mutate the .of file using pybFoam (as a user would edit the case)
3. Reload and verify that validation errors are captured correctly

All errors go through the staged workflow (validate=False + check_validation()
or collect_errors()), matching how the incompressibleFluid solver uses configs.
"""

import pybFoam as pyf
import pytest

from neofoam.io import BaseConfig, OF, IOStrategy
from neofoam.schemes import (
    DdtScheme,
    GradScheme,
    DivScheme,
    LaplacianScheme,
    Backward,
    SteadyState,
    LeastSquaresGrad,
)


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------


@IOStrategy(OF("fvSchemes.of", subdict="U"))
class FvSchemesIntegConfig_U(BaseConfig):
    ddt: DdtScheme
    grad: GradScheme
    div: DivScheme
    laplacian: LaplacianScheme


@IOStrategy(OF("fvSchemes.of", subdict="p"))
class FvSchemesIntegConfig_p(BaseConfig):
    laplacian: LaplacianScheme


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _set_scheme_value(of_file_path, subdict_name: str, key: str, value: str):
    """Mutate a single key in an OF dictionary file using pybFoam."""
    d = pyf.dictionary.read(str(of_file_path))
    sub = d.subDict(subdict_name)
    sub.set(key, value)
    d.write(str(of_file_path))


# ---------------------------------------------------------------------------
# Integration tests: valid mutations reload correctly
# ---------------------------------------------------------------------------


def test_mutate_ddt_to_backward(temp_fixture_copy):
    """Change ddt to 'backward' via pybFoam, reload and verify Backward type."""
    test_file = temp_fixture_copy("fvSchemes.of")
    _set_scheme_value(test_file, "U", "ddt", "backward")

    cfg = FvSchemesIntegConfig_U.load(case_dir=test_file.parent)
    assert isinstance(cfg.ddt, Backward)


def test_mutate_ddt_to_steadyState(temp_fixture_copy):
    """Change ddt to 'steadyState' via pybFoam, reload and verify SteadyState."""
    test_file = temp_fixture_copy("fvSchemes.of")
    _set_scheme_value(test_file, "U", "ddt", "steadyState")

    cfg = FvSchemesIntegConfig_U.load(case_dir=test_file.parent)
    assert isinstance(cfg.ddt, SteadyState)


def test_mutate_grad_to_leastSquares(temp_fixture_copy):
    """Change grad to 'pointCellsLeastSquares', reload and verify type."""
    test_file = temp_fixture_copy("fvSchemes.of")
    _set_scheme_value(test_file, "U", "grad", "pointCellsLeastSquares")

    cfg = FvSchemesIntegConfig_U.load(case_dir=test_file.parent)
    assert isinstance(cfg.grad, LeastSquaresGrad)


# ---------------------------------------------------------------------------
# Integration tests: invalid mutations captured by staged workflow
# ---------------------------------------------------------------------------


def test_mutate_ddt_invalid_staged_workflow(temp_fixture_copy):
    """Invalid ddt via pybFoam mutation: check_validation() captures the error."""
    test_file = temp_fixture_copy("fvSchemes.of")
    _set_scheme_value(test_file, "U", "ddt", "InvalidScheme")

    cfg = FvSchemesIntegConfig_U.load(case_dir=test_file.parent, validate=False)
    errors = cfg.check_validation()

    assert len(errors) >= 1
    ddt_errors = [e for e in errors if "ddt" in str(e.field)]
    assert len(ddt_errors) >= 1
    assert ddt_errors[0].file_name == "fvSchemes.of"
    assert ddt_errors[0].subdict == "U"


def test_mutate_grad_invalid_staged_workflow(temp_fixture_copy):
    """Invalid grad via pybFoam mutation: check_validation() captures the error."""
    test_file = temp_fixture_copy("fvSchemes.of")
    _set_scheme_value(test_file, "U", "grad", "invalidGrad")

    cfg = FvSchemesIntegConfig_U.load(case_dir=test_file.parent, validate=False)
    errors = cfg.check_validation()

    assert len(errors) >= 1
    grad_errors = [e for e in errors if "grad" in str(e.field)]
    assert len(grad_errors) >= 1


def test_mutate_laplacian_invalid_collect_errors(temp_fixture_copy):
    """Invalid laplacian via pybFoam: collect_errors() returns structured errors."""
    test_file = temp_fixture_copy("fvSchemes.of")
    _set_scheme_value(test_file, "U", "laplacian", "invalid linear corrected")

    errors = FvSchemesIntegConfig_U.collect_errors(case_dir=test_file.parent)

    assert len(errors) >= 1
    lap_errors = [e for e in errors if "laplacian" in str(e.field)]
    assert len(lap_errors) >= 1
    assert all(e.subdict == "U" for e in errors)


def test_mutate_multiple_schemes_all_errors_captured(temp_fixture_copy):
    """Multiple invalid mutations: all errors captured in a single check_validation()."""
    test_file = temp_fixture_copy("fvSchemes.of")
    _set_scheme_value(test_file, "U", "ddt", "InvalidDdt")
    _set_scheme_value(test_file, "U", "grad", "invalidGrad")

    cfg = FvSchemesIntegConfig_U.load(case_dir=test_file.parent, validate=False)
    errors = cfg.check_validation()

    assert len(errors) >= 2
    error_fields = [str(e.field) for e in errors]
    assert any("ddt" in f for f in error_fields)
    assert any("grad" in f for f in error_fields)


# ---------------------------------------------------------------------------
# Integration tests: valid file round-trip
# ---------------------------------------------------------------------------


def test_valid_file_staged_workflow_no_errors(temp_fixture_copy):
    """Staged workflow on valid fvSchemes.of returns no errors."""
    temp_fixture_copy("fvSchemes.of")
    test_dir = temp_fixture_copy("fvSchemes.of").parent

    cfg = FvSchemesIntegConfig_U.load(case_dir=test_dir, validate=False)
    errors = cfg.check_validation()
    assert errors == []


def test_collect_errors_valid_returns_empty(temp_fixture_copy):
    """collect_errors() returns [] for a valid fvSchemes.of."""
    test_file = temp_fixture_copy("fvSchemes.of")

    errors = FvSchemesIntegConfig_U.collect_errors(case_dir=test_file.parent)
    assert errors == []


def test_mutate_p_laplacian_invalid(temp_fixture_copy):
    """Mutation of p laplacian subdict is captured with correct subdict metadata."""
    test_file = temp_fixture_copy("fvSchemes.of")
    _set_scheme_value(test_file, "p", "laplacian", "invalid linear corrected")

    errors = FvSchemesIntegConfig_p.collect_errors(case_dir=test_file.parent)
    assert len(errors) >= 1
    assert all(e.subdict == "p" for e in errors)
