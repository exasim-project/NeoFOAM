# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Integration tests for validation functionality in 3-stage initialization.

Tests validation of LOAD and RESOLVE stages using the actual DummySolver
implementation with real configuration files. Configuration classes use
the @IOStrategy(YAML(...)) decorator pattern for declarative IO configuration.
"""

import shutil
from pathlib import Path

import pytest

from .dummy_init import create_init


# ============================================================================
# Fixtures: Setup configuration directory
# ============================================================================


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory with copies of valid configs."""
    configs_src = Path(__file__).parent / "configs"
    configs_dst = tmp_path / "configs"

    # Copy valid config files
    shutil.copytree(configs_src, configs_dst)

    return configs_dst


# ============================================================================
# Test: Valid configuration
# ============================================================================


def test_dummy_solver_valid_configs():
    """Integration test: successful initialization with default valid configs."""
    init = create_init()
    ctx = init.run()

    assert ctx is not None
    assert "field1" in ctx.fields
    assert "field2" in ctx.fields
    assert ctx.models["config"]["param1"] == 1e-05


def test_dummy_solver_invalid_config(temp_config_dir, monkeypatch):
    """Integration test: validation fails with all invalid configs from configs_invalid folder."""
    # Copy all invalid configs from configs_invalid folder
    invalid_configs_dir = Path(__file__).parent / "configs_invalid"

    # Map of invalid config files to their target names
    invalid_configs = {
        "algorithm_config_invalid_param1.yaml": "algorithm_config.yaml",
        "mesh_config_invalid_npoints.yaml": "mesh_config.yaml",
        "model1_config_invalid_prop1.yaml": "model1_config.yaml",
        "solver_config_invalid_param1.yaml": "solver_config.yaml",
    }

    for invalid_file, target_file in invalid_configs.items():
        src = invalid_configs_dir / invalid_file
        dst = temp_config_dir / target_file
        shutil.copy(src, dst)

    init = create_init(case_dir=temp_config_dir)
    loading_results = init.run_load()
    val_errors = loading_results.validate()

    # Sort errors by file_name and field for deterministic testing
    val_errors_sorted = sorted(val_errors, key=lambda e: (e.file_name, str(e.field)))

    # Should have at least 4 validation errors from different files
    assert len(val_errors_sorted) >= 4

    # Extract errors by file
    algo_errors = [
        e for e in val_errors_sorted if e.file_name == "algorithm_config.yaml"
    ]
    mesh_errors = [e for e in val_errors_sorted if e.file_name == "mesh_config.yaml"]
    model1_errors = [
        e for e in val_errors_sorted if e.file_name == "model1_config.yaml"
    ]
    solver_errors = [
        e for e in val_errors_sorted if e.file_name == "solver_config.yaml"
    ]

    # Check algorithm_config.yaml errors
    assert len(algo_errors) >= 1, "Expected at least 1 error in algorithm_config.yaml"
    algo_error = algo_errors[0]
    assert algo_error.file_name == "algorithm_config.yaml"
    assert "param1" in str(algo_error.field)
    assert algo_error.input_value == -0.00001
    assert algo_error.error_type == "greater_than"

    # Check mesh_config.yaml errors
    assert len(mesh_errors) >= 1, "Expected at least 1 error in mesh_config.yaml"
    mesh_error = mesh_errors[0]
    assert mesh_error.file_name == "mesh_config.yaml"
    assert "nPoints" in str(mesh_error.field)
    assert mesh_error.input_value == -1
    assert mesh_error.error_type == "greater_than"

    # Check model1_config.yaml errors
    assert len(model1_errors) >= 1, "Expected at least 1 error in model1_config.yaml"
    model1_error = model1_errors[0]
    assert model1_error.file_name == "model1_config.yaml"
    assert "prop1" in str(model1_error.field)
    assert model1_error.input_value == 1.5
    assert model1_error.error_type == "less_than_equal"

    # Check solver_config.yaml errors (has multiple: param1, endTime)
    assert len(solver_errors) >= 2, (
        f"Expected at least 2 errors in solver_config.yaml, got {len(solver_errors)}"
    )

    # Find param1 and endTime errors
    param1_error = next((e for e in solver_errors if "param1" in str(e.field)), None)
    endtime_error = next((e for e in solver_errors if "endTime" in str(e.field)), None)

    assert param1_error is not None, "Missing param1 error in solver_config.yaml"
    assert param1_error.file_name == "solver_config.yaml"
    assert param1_error.input_value == -1.0
    assert param1_error.error_type == "greater_than"

    assert endtime_error is not None, "Missing endTime error in solver_config.yaml"
    assert endtime_error.file_name == "solver_config.yaml"
    assert endtime_error.input_value == -1.0

    # Verify all errors have proper metadata
    for err in val_errors_sorted:
        assert err.file_name, f"Error missing file_name: {err}"
        assert err.file_name.endswith(".yaml"), f"Invalid file_name: {err.file_name}"
