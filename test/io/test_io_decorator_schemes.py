# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Unit tests for OpenFOAMStrategy loading discriminated union scheme types.

Demonstrates:
- Loading configs with DdtScheme, GradScheme, DivScheme, LaplacianScheme
  from an OpenFOAM dictionary file via OF()
- The BeforeValidator on each scheme type parses raw strings into typed models
- Validation errors from unknown scheme strings are captured correctly
- Integration with the staged workflow (validate=False + check_validation())
"""

import shutil

import pytest
from pydantic import ValidationError

from neofoam.io import BaseConfig, OF, IOStrategy
from neofoam.schemes import (
    DdtScheme,
    GradScheme,
    DivScheme,
    LaplacianScheme,
    Euler,
    GaussGrad,
    GaussDiv,
    GaussLaplacian,
    Linear,
    LinearUpwind,
    Corrected,
    Backward,
)


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------


@IOStrategy(OF("fvSchemes.of", subdict="U"))
class FvSchemesConfig_U(BaseConfig):
    """Discretization schemes for U field, loaded from an OF dictionary."""

    ddt: DdtScheme
    grad: GradScheme
    div: DivScheme
    laplacian: LaplacianScheme


@IOStrategy(OF("fvSchemes.of", subdict="p"))
class FvSchemesConfig_p(BaseConfig):
    """Discretization schemes for p field, loaded from an OF dictionary."""

    laplacian: LaplacianScheme


# ---------------------------------------------------------------------------
# Unit tests: load valid fvSchemes.of
# ---------------------------------------------------------------------------


def test_load_fvSchemes_ddt(io_fixtures):
    """DdtScheme field is parsed from plain string 'Euler' into Euler instance."""
    cfg = FvSchemesConfig_U.load(case_dir=io_fixtures)
    assert isinstance(cfg.ddt, Euler)


def test_load_fvSchemes_grad(io_fixtures):
    """GradScheme field 'Gauss linear' is parsed into GaussGrad with Linear."""
    cfg = FvSchemesConfig_U.load(case_dir=io_fixtures)
    assert isinstance(cfg.grad, GaussGrad)
    assert isinstance(cfg.grad.interpolation, Linear)


def test_load_fvSchemes_div(io_fixtures):
    """DivScheme field 'Gauss linearUpwind grad(U)' is parsed into GaussDiv."""
    cfg = FvSchemesConfig_U.load(case_dir=io_fixtures)
    assert isinstance(cfg.div, GaussDiv)
    assert isinstance(cfg.div.interpolation, LinearUpwind)
    assert cfg.div.interpolation.grad_field == "grad(U)"


def test_load_fvSchemes_laplacian(io_fixtures):
    """LaplacianScheme field 'Gauss linear corrected' is parsed correctly."""
    cfg = FvSchemesConfig_U.load(case_dir=io_fixtures)
    assert isinstance(cfg.laplacian, GaussLaplacian)
    assert isinstance(cfg.laplacian.interpolation, Linear)
    assert isinstance(cfg.laplacian.sn_grad, Corrected)


def test_load_fvSchemes_p(io_fixtures):
    """p pressure scheme config loads from 'p' subdict correctly."""
    cfg = FvSchemesConfig_p.load(case_dir=io_fixtures)
    assert isinstance(cfg.laplacian, GaussLaplacian)
    assert isinstance(cfg.laplacian.interpolation, Linear)
    assert isinstance(cfg.laplacian.sn_grad, Corrected)


def test_staged_load_valid_returns_no_errors(io_fixtures):
    """validate=False then check_validation() returns no errors for valid fvSchemes."""
    cfg = FvSchemesConfig_U.load(case_dir=io_fixtures, validate=False)
    errors = cfg.check_validation()
    assert errors == []


# ---------------------------------------------------------------------------
# Unit tests: invalid scheme strings
# ---------------------------------------------------------------------------


def test_load_invalid_ddt_raises_with_validate_true(io_fixtures):
    """Loading invalid ddt scheme with validate=True raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        FvSchemesConfig_U.load(
            case_dir=io_fixtures,
            validate=True,
            file="invalid_fvSchemes.of",
        )
    errors = exc_info.value.errors()
    ddt_errors = [e for e in errors if "ddt" in str(e["loc"])]
    assert len(ddt_errors) >= 1


def test_staged_invalid_ddt_check_validation(io_fixtures):
    """Staged load of invalid ddt: validate=False then check_validation() catches it."""
    cfg = FvSchemesConfig_U.load(
        case_dir=io_fixtures,
        validate=False,
        file="invalid_fvSchemes.of",
    )
    errors = cfg.check_validation()
    assert len(errors) >= 1
    ddt_errors = [e for e in errors if "ddt" in str(e.field)]
    assert len(ddt_errors) >= 1
    assert ddt_errors[0].file_name == "fvSchemes.of"


def test_collect_errors_returns_structured_errors(io_fixtures):
    """collect_errors() returns ValidationErrors for invalid ddt scheme."""
    errors = FvSchemesConfig_U.collect_errors(
        case_dir=io_fixtures,
        file="invalid_fvSchemes.of",
    )
    assert len(errors) >= 1
    assert all(e.subdict == "U" for e in errors)


# ---------------------------------------------------------------------------
# Unit tests: save / roundtrip
# ---------------------------------------------------------------------------


def test_model_dump_produces_scheme_strings(io_fixtures):
    """model_dump(mode='python') on a loaded fvSchemes config yields OF strings."""
    cfg = FvSchemesConfig_U.load(case_dir=io_fixtures)
    data = cfg.model_dump(mode="python")
    assert data["ddt"] == "Euler"
    assert data["grad"] == "Gauss linear"
    assert data["div"] == "Gauss linearUpwind grad(U)"
    assert data["laplacian"] == "Gauss linear corrected"


def test_save_fvSchemes_roundtrip_U(io_fixtures, tmp_path):
    """Load FvSchemesConfig_U, save in-place, reload — all fields reproduced."""
    shutil.copy(io_fixtures / "fvSchemes.of", tmp_path / "fvSchemes.of")
    cfg = FvSchemesConfig_U.load(case_dir=tmp_path)
    cfg.save(case_dir=tmp_path)
    cfg2 = FvSchemesConfig_U.load(case_dir=tmp_path)
    assert isinstance(cfg2.ddt, Euler)
    assert isinstance(cfg2.grad, GaussGrad)
    assert isinstance(cfg2.div, GaussDiv)
    assert isinstance(cfg2.laplacian, GaussLaplacian)
    assert isinstance(cfg2.laplacian.interpolation, Linear)
    assert isinstance(cfg2.laplacian.sn_grad, Corrected)


def test_save_fvSchemes_roundtrip_p(io_fixtures, tmp_path):
    """FvSchemesConfig_p save/reload roundtrip preserves laplacian scheme."""
    shutil.copy(io_fixtures / "fvSchemes.of", tmp_path / "fvSchemes.of")
    cfg = FvSchemesConfig_p.load(case_dir=tmp_path)
    cfg.save(case_dir=tmp_path)
    cfg2 = FvSchemesConfig_p.load(case_dir=tmp_path)
    assert isinstance(cfg2.laplacian, GaussLaplacian)
    assert isinstance(cfg2.laplacian.interpolation, Linear)
    assert isinstance(cfg2.laplacian.sn_grad, Corrected)


def test_save_fvSchemes_modified_ddt_persists(io_fixtures, tmp_path):
    """After changing ddt to Backward and saving, reload reflects the change."""
    shutil.copy(io_fixtures / "fvSchemes.of", tmp_path / "fvSchemes.of")
    cfg = FvSchemesConfig_U.load(case_dir=tmp_path)
    cfg.ddt = Backward()
    cfg.save(case_dir=tmp_path)
    cfg2 = FvSchemesConfig_U.load(case_dir=tmp_path)
    assert isinstance(cfg2.ddt, Backward)
