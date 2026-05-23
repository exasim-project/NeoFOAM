# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Integration tests for validation functionality in 3-stage initialization.

Tests validation of LOAD and RESOLVE stages using the actual DummySolver
implementation with real configuration files. Configuration classes use
the @IOStrategy(YAML(...)) decorator pattern for declarative IO configuration.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from .dummy_init import create_init


# ============================================================================
# Fixtures: Setup configuration directory
# ============================================================================


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create temporary config directory with copies of valid configs."""
    configs_src = Path(__file__).parent / "configs"
    configs_dst = tmp_path / "configs"

    # Copy valid config files
    shutil.copytree(configs_src, configs_dst)

    return configs_dst


# ============================================================================
# Test: Valid configuration
# ============================================================================


def test_dummy_solver_valid_configs() -> None:
    """Integration test: successful initialization with default valid configs."""
    init = create_init()
    ctx = init.run()

    assert ctx is not None
    assert "field1" in ctx.fields
    assert "field2" in ctx.fields
    assert ctx.models["config"]["param1"] == 1e-05


def test_dummy_solver_invalid_config(temp_config_dir: Path, monkeypatch: Any) -> None:
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

    # Expected per-file error specifications:
    #   (file_name, min_errors, [(field_substr, input_value, error_type), ...])
    expected: list[tuple[str, int, list[tuple[str, float, str | None]]]] = [
        ("algorithm_config.yaml", 1, [("param1", -0.00001, "greater_than")]),
        ("mesh_config.yaml", 1, [("nPoints", -1, "greater_than")]),
        ("model1_config.yaml", 1, [("prop1", 1.5, "less_than_equal")]),
        (
            "solver_config.yaml",
            2,
            [("param1", -1.0, "greater_than"), ("endTime", -1.0, None)],
        ),
    ]

    for file_name, min_count, field_checks in expected:
        file_errors = [e for e in val_errors_sorted if e.file_name == file_name]
        assert len(file_errors) >= min_count, (
            f"Expected at least {min_count} error(s) in {file_name}, got {len(file_errors)}"
        )

        for field_substr, input_value, error_type in field_checks:
            match = next((e for e in file_errors if field_substr in str(e.field)), None)
            assert match is not None, f"Missing '{field_substr}' error in {file_name}"
            assert match.input_value == input_value, (
                f"{file_name}: expected input_value={input_value} for '{field_substr}', "
                f"got {match.input_value}"
            )
            if error_type is not None:
                assert match.error_type == error_type, (
                    f"{file_name}: expected error_type='{error_type}' for '{field_substr}', "
                    f"got '{match.error_type}'"
                )

    # Verify all errors have proper metadata
    for err in val_errors_sorted:
        assert err.file_name, f"Error missing file_name: {err}"
        assert err.file_name.endswith(".yaml"), f"Invalid file_name: {err.file_name}"
