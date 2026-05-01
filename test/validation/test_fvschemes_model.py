# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Unit tests for FvSchemesConfig — load, parse, roundtrip.

Verifies that the Pydantic model is correctly built from an OpenFOAM fvSchemes
file, that scheme values parse into typed models, and that save/reload roundtrips.
"""

import shutil
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from neofoam.foam.fv_configs import FvSchemesConfig
from neofoam.foam.schemes import (
    Corrected,
    DdtScheme,
    DivScheme,
    Euler,
    GaussDiv,
    GaussGrad,
    GaussLaplacian,
    GradScheme,
    InterpolationScheme,
    LaplacianScheme,
    Linear,
    SnGradScheme,
)

FIXTURE = Path(__file__).parent / "fvSchemes.of"


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    """Create a minimal case dir with system/fvSchemes from the fixture."""
    system = tmp_path / "system"
    system.mkdir()
    shutil.copy(FIXTURE, system / "fvSchemes")
    return tmp_path


# ============================================================================
# Load — structure
# ============================================================================


def test_load_has_all_sections(case_dir: Path) -> None:
    cfg = FvSchemesConfig.load(case_dir=case_dir, validate=False)
    data = cfg.model_dump()
    for section in [
        "ddtSchemes",
        "gradSchemes",
        "divSchemes",
        "laplacianSchemes",
        "interpolationSchemes",
        "snGradSchemes",
        "wallDist",
    ]:
        assert section in data, f"Missing section: {section}"


def test_load_concrete_entries(case_dir: Path) -> None:
    data = FvSchemesConfig.load(case_dir=case_dir, validate=False).model_dump()
    assert data["ddtSchemes"]["default"] == "Euler"
    assert data["gradSchemes"]["grad(U)"] == "Gauss linear"
    assert data["gradSchemes"]["grad(p)"] == "Gauss linear"
    assert data["divSchemes"]["div(phi,U)"] == "Gauss linearUpwind grad(U)"
    assert data["divSchemes"]["div(phi,nuTilda)"] == "Gauss upwind"
    assert data["laplacianSchemes"]["laplacian(nuEff,U)"] == "Gauss linear corrected"
    assert data["laplacianSchemes"]["laplacian(rAU,p)"] == "Gauss linear corrected"
    assert data["interpolationSchemes"]["flux(HbyA)"] == "linear"
    assert data["interpolationSchemes"]["interpolate(rAU)"] == "linear"
    assert data["snGradSchemes"]["snGrad(p)"] == "corrected"
    assert data["wallDist"]["method"] == "meshWave"


# ============================================================================
# Parse — scheme values become typed Pydantic models
# ============================================================================


@pytest.mark.parametrize(
    "raw,scheme_type,expected_cls",
    [
        ("Euler", DdtScheme, Euler),
        ("Gauss linear", GradScheme, GaussGrad),
        ("Gauss linearUpwind grad(U)", DivScheme, GaussDiv),
        ("Gauss upwind", DivScheme, GaussDiv),
        ("Gauss linear corrected", LaplacianScheme, GaussLaplacian),
        ("corrected", SnGradScheme, Corrected),
        ("linear", InterpolationScheme, Linear),
    ],
    ids=[
        "ddt-Euler",
        "grad-Gauss",
        "div-linearUpwind",
        "div-upwind",
        "laplacian",
        "snGrad",
        "interp",
    ],
)
def test_scheme_values_parse(raw: str, scheme_type: type, expected_cls: type) -> None:
    parsed = TypeAdapter(scheme_type).validate_python(raw)
    assert isinstance(parsed, expected_cls)


# ============================================================================
# Roundtrip — typed scheme serialization preserves the original string
# ============================================================================


@pytest.mark.parametrize(
    "scheme_type,raw_string",
    [
        (DdtScheme, "Euler"),
        (GradScheme, "Gauss linear"),
        (DivScheme, "Gauss linearUpwind grad(U)"),
        (DivScheme, "Gauss upwind"),
        (LaplacianScheme, "Gauss linear corrected"),
        (SnGradScheme, "corrected"),
        (InterpolationScheme, "linear"),
    ],
    ids=[
        "ddt",
        "grad",
        "div-linearUpwind",
        "div-upwind",
        "laplacian",
        "snGrad",
        "interp",
    ],
)
def test_scheme_roundtrip_serialization(scheme_type: type, raw_string: str) -> None:
    parsed = TypeAdapter(scheme_type).validate_python(raw_string)
    serialized = parsed.model_dump(mode="python")
    assert serialized == raw_string


# ============================================================================
# Roundtrip — save and reload produces same data
# ============================================================================


@pytest.mark.xfail(
    reason="OpenFOAMStrategy._write_fields does not write extra fields for extra='allow' models"
)
def test_roundtrip_save_reload(case_dir: Path, tmp_path: Path) -> None:
    cfg = FvSchemesConfig.load(case_dir=case_dir, validate=False)
    original = cfg.model_dump()

    # Save to a new location
    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    (save_dir / "system").mkdir()
    cfg.save(case_dir=save_dir)

    # Reload and compare
    cfg2 = FvSchemesConfig.load(case_dir=save_dir, validate=False)
    reloaded = cfg2.model_dump()
    assert original == reloaded


# ============================================================================
# Build typed Pydantic model from requirements
# ============================================================================

from pydantic import ValidationError

from neofoam.foam.requirements import SchemeRequirement
from neofoam.foam.verification import (
    _sanitize_field_name,
    build_scheme_model,
    flatten_fvschemes,
)


def test_build_scheme_model_has_typed_fields() -> None:
    reqs = [
        SchemeRequirement("ddtSchemes", "ddt(U)"),
        SchemeRequirement("divSchemes", "div(phi,U)"),
        SchemeRequirement("gradSchemes", "grad(U)"),
    ]
    Model = build_scheme_model(reqs)
    schema = Model.model_json_schema()
    assert "ddtSchemes_ddt_U" in schema["properties"]
    assert "divSchemes_div_phi_U" in schema["properties"]
    assert "gradSchemes_grad_U" in schema["properties"]


def test_build_scheme_model_validates_values() -> None:
    reqs = [SchemeRequirement("ddtSchemes", "ddt(U)")]
    Model = build_scheme_model(reqs)

    # Valid value
    instance = Model(ddtSchemes_ddt_U="Euler")
    assert instance.ddtSchemes_ddt_U.type == "Euler"  # type: ignore[union-attr]

    # Invalid value
    with pytest.raises(ValidationError):
        Model(ddtSchemes_ddt_U="invalidScheme")


def test_build_scheme_model_unknown_section_gets_str() -> None:
    """Sections not in SECTION_TO_TYPE (e.g., wallDist) get str type."""
    reqs = [SchemeRequirement("wallDist", "method")]
    Model = build_scheme_model(reqs)
    instance = Model(wallDist_method="meshWave")
    assert instance.wallDist_method == "meshWave"  # type: ignore[attr-defined]


def test_flatten_fvschemes_extracts_values() -> None:
    fv_data = {
        "ddtSchemes": {"ddt(U)": "Euler"},
        "divSchemes": {"default": "none", "div(phi,U)": "Gauss upwind"},
    }
    reqs = [
        SchemeRequirement("ddtSchemes", "ddt(U)"),
        SchemeRequirement("divSchemes", "div(phi,U)"),
    ]
    flat = flatten_fvschemes(fv_data, reqs)
    assert flat["ddtSchemes_ddt_U"] == "Euler"
    assert flat["divSchemes_div_phi_U"] == "Gauss upwind"


def test_flatten_fvschemes_falls_back_to_default() -> None:
    fv_data = {"ddtSchemes": {"default": "Euler"}}
    reqs = [SchemeRequirement("ddtSchemes", "ddt(U)")]
    flat = flatten_fvschemes(fv_data, reqs)
    assert flat["ddtSchemes_ddt_U"] == "Euler"


def test_flatten_fvschemes_ignores_default_none() -> None:
    fv_data = {"divSchemes": {"default": "none"}}
    reqs = [SchemeRequirement("divSchemes", "div(phi,U)")]
    flat = flatten_fvschemes(fv_data, reqs)
    assert "divSchemes_div_phi_U" not in flat


def test_build_and_validate_from_file(case_dir: Path) -> None:
    """Load fvSchemes, build model from requirements, validate end-to-end."""
    cfg = FvSchemesConfig.load(case_dir=case_dir, validate=False)
    fv_data = cfg.model_dump()

    reqs = [
        SchemeRequirement("ddtSchemes", "ddt(U)"),  # has "default Euler" → falls back
        SchemeRequirement("divSchemes", "div(phi,U)"),
        SchemeRequirement("gradSchemes", "grad(U)"),
        SchemeRequirement("laplacianSchemes", "laplacian(nuEff,U)"),
        SchemeRequirement("snGradSchemes", "snGrad(p)"),
        SchemeRequirement("interpolationSchemes", "flux(HbyA)"),
        SchemeRequirement("wallDist", "method"),
    ]

    Model = build_scheme_model(reqs)
    flat = flatten_fvschemes(fv_data, reqs)
    instance = Model(**flat)

    # Typed fields parsed correctly
    assert instance.ddtSchemes_ddt_U.type == "Euler"  # type: ignore[union-attr]
    assert instance.wallDist_method == "meshWave"  # type: ignore[attr-defined]


def test_json_schema_has_descriptions() -> None:
    reqs = [
        SchemeRequirement("ddtSchemes", "ddt(U)"),
        SchemeRequirement("divSchemes", "div(phi,U)"),
    ]
    Model = build_scheme_model(reqs)
    schema = Model.model_json_schema()
    for req in reqs:
        name = _sanitize_field_name(req.section, req.key)
        prop = schema["properties"][name]
        assert f"{req.section}.{req.key}" in prop["description"]
