# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Integration tests for turbulence models with real OpenFOAM test cases."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (@Model.build decorator removed)"
)

from pathlib import Path
import shutil

from foamadapter.turbulence import TurbulenceModel


@pytest.fixture
def turb_case_dir():
    """Get the turbulence test case directory."""
    test_dir = Path(__file__).parent
    case_dir = test_dir / "turb_case"

    if not case_dir.exists():
        pytest.skip(f"Test case directory not found: {case_dir}")

    return case_dir


@pytest.fixture
def temp_turb_case(turb_case_dir, tmp_path):
    """Create a temporary copy of turb_case for modification."""
    temp_case = tmp_path / "turb_case_temp"
    shutil.copytree(turb_case_dir, temp_case)
    return temp_case


def create_turbulence_properties(case_dir, simulation_type, model_type):
    """Create turbulenceProperties file with specified model."""
    turb_props = case_dir / "constant" / "turbulenceProperties"
    turb_props.parent.mkdir(parents=True, exist_ok=True)

    if simulation_type == "RAS":
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType      RAS;

RAS
{{
    RASModel        {model_type};
    turbulence      on;
    printCoeffs     on;
}}

// ************************************************************************* //
"""
    elif simulation_type == "LES":
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType      LES;

LES
{{
    LESModel        {model_type};
    turbulence      on;
    printCoeffs     on;
}}

// ************************************************************************* //
"""
    else:  # laminar
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType      laminar;

// ************************************************************************* //
"""

    turb_props.write_text(content)
    return turb_props


@pytest.mark.parametrize(
    "simulation_type,model_type,expected_class,expected_key",
    [
        ("RAS", "kOmegaSST", "kOmegaSSTModel", "RAS_kOmegaSST"),
        ("RAS", "kEpsilon", "kEpsilonModel", "RAS_kEpsilon"),
        ("LES", "Smagorinsky", "SmagorinskyModel", "LES_Smagorinsky"),
        ("laminar", None, "LaminarModel", "laminar"),
    ],
)
def test_turbulence_model_from_file(
    turb_case_dir, simulation_type, model_type, expected_class, expected_key
):
    """Test reading different turbulence models from file by modifying real case."""
    # Read the original turbulenceProperties to preserve exact format
    turb_props = turb_case_dir / "constant" / "turbulenceProperties"
    original_content = turb_props.read_text()

    try:
        # Create modified content with new model
        if simulation_type == "RAS":
            # Replace the RAS model type
            import re

            new_content = re.sub(
                r"RASModel\s+\w+;", f"RASModel        {model_type};", original_content
            )
        elif simulation_type == "LES":
            # Convert to LES
            new_content = original_content.replace(
                "simulationType      RAS;", "simulationType      LES;"
            )
            new_content = new_content.replace("RAS\n{", "LES\n{")
            new_content = new_content.replace("RASModel", "LESModel")
            import re

            new_content = re.sub(
                r"LESModel\s+\w+;", f"LESModel        {model_type};", new_content
            )
        else:  # laminar
            new_content = original_content.replace(
                "simulationType      RAS;", "simulationType      laminar;"
            )
            # Remove RAS section
            import re

            new_content = re.sub(r"RAS\s*\{[^}]*\}", "", new_content, flags=re.DOTALL)

        turb_props.write_text(new_content)

        # Read and validate - pybFoam needs to be run from case directory
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(turb_case_dir)
            wrapper = TurbulenceModel.from_file(str(turb_props))
            model = wrapper.config

            assert model.__class__.__name__ == expected_class
            assert model.model_key == expected_key
            assert model.simulation_type == simulation_type
            assert model.turb_model_type == model_type
            assert hasattr(model, "setup")
            assert hasattr(model, "operations")
        finally:
            os.chdir(original_dir)
    finally:
        # Restore original content
        turb_props.write_text(original_content)


def test_real_case_kEpsilon(turb_case_dir):
    """Test that the real turb_case with kEpsilon now works with composite key."""
    turb_props = turb_case_dir / "constant" / "turbulenceProperties"

    if not turb_props.exists():
        pytest.skip(f"turbulenceProperties not found: {turb_props}")

    # The real case uses kEpsilon, which now works with model_key discriminator
    wrapper = TurbulenceModel.from_file(str(turb_props))
    model = wrapper.config

    assert model.__class__.__name__ == "kEpsilonModel"
    assert model.model_key == "RAS_kEpsilon"
    assert model.simulation_type == "RAS"
    assert model.turb_model_type == "kEpsilon"


@pytest.mark.parametrize("model_type", ["kOmegaSST", "Smagorinsky"])
def test_integration_with_operations(turb_case_dir, model_type):
    """Test full integration: read model, setup, and collect operations."""
    simulation_type = "RAS" if model_type == "kOmegaSST" else "LES"

    # Read the original turbulenceProperties to preserve exact format
    turb_props = turb_case_dir / "constant" / "turbulenceProperties"
    original_content = turb_props.read_text()

    try:
        # Create modified content with new model
        if simulation_type == "RAS":
            import re

            new_content = re.sub(
                r"RASModel\s+\w+;", f"RASModel        {model_type};", original_content
            )
        else:  # LES
            new_content = original_content.replace(
                "simulationType      RAS;", "simulationType      LES;"
            )
            new_content = new_content.replace("RAS\n{", "LES\n{")
            new_content = new_content.replace("RASModel", "LESModel")
            import re

            new_content = re.sub(
                r"LESModel\s+\w+;", f"LESModel        {model_type};", new_content
            )

        turb_props.write_text(new_content)

        # Read model - pybFoam needs to be run from case directory
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(turb_case_dir)
            wrapper = TurbulenceModel.from_file(str(turb_props))
            model = wrapper.config
        finally:
            os.chdir(original_dir)
    finally:
        # Restore original content
        turb_props.write_text(original_content)

    # Test setup returns LazyInit
    from unittest.mock import MagicMock

    mock_mesh = MagicMock()
    lazy_inits = model.setup(mock_mesh)

    assert len(lazy_inits) == 1
    assert lazy_inits[0].name == "models.turbulence"
    assert lazy_inits[0].depends_on == [
        "fields.U",
        "fields.phi",
        "fields.laminarTransport",
    ]

    # Test operations collection
    ops = model.operations()
    assert ops is not None
    assert ops is not None
