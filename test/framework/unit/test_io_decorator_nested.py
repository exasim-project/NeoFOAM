# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for nested subdict IO patterns.

Demonstrates:
- Loading from flat subdicts (top-level keys)
- Loading from nested subdicts (dot notation)
- Subdict isolation (each config only sees its own section)
"""

import pytest

from foamadapter.io import (
    BaseConfig,
    YAML,
    JSON,
    IOStrategy,
)


# ============================================================================
# Config Classes
# ============================================================================


@IOStrategy(YAML("nested.yaml", subdict="application"))
class ApplicationYAMLConfig(BaseConfig):
    name: str
    version: str


@IOStrategy(YAML("nested.yaml", subdict="solver.linear"))
class LinearSolverYAMLConfig(BaseConfig):
    method: str
    tolerance: float
    maxIterations: int


@IOStrategy(YAML("nested.yaml", subdict="solver.nonlinear"))
class NonlinearSolverYAMLConfig(BaseConfig):
    method: str
    tolerance: float
    maxIterations: int


@IOStrategy(JSON("nested.json", subdict="application"))
class ApplicationJSONConfig(BaseConfig):
    name: str
    version: str


@IOStrategy(JSON("nested.json", subdict="solver.linear"))
class LinearSolverJSONConfig(BaseConfig):
    method: str
    tolerance: float
    maxIterations: int


@IOStrategy(JSON("nested.json", subdict="solver.nonlinear"))
class NonlinearSolverJSONConfig(BaseConfig):
    method: str
    tolerance: float
    maxIterations: int


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize(
    "app_class,linear_class,nonlinear_class",
    [
        (ApplicationYAMLConfig, LinearSolverYAMLConfig, NonlinearSolverYAMLConfig),
        (ApplicationJSONConfig, LinearSolverJSONConfig, NonlinearSolverJSONConfig),
    ],
)
def test_load_nested(io_fixtures, app_class, linear_class, nonlinear_class):
    """Test loading multiple nested subdicts from fixture (YAML and JSON).

    Demonstrates:
    - Flat subdict: application (top-level key)
    - Nested subdict: solver.linear (2-level nesting)
    - Nested subdict: solver.nonlinear (2-level nesting, different key)
    - Subdict isolation (each config only sees its own section)
    """
    # Load each subdict independently
    app = app_class.load(io_fixtures)
    linear = linear_class.load(io_fixtures)
    nonlinear = nonlinear_class.load(io_fixtures)

    # Verify isolation - each config only sees its subdict
    assert app.name == "solver"
    assert app.version == "1.0"

    assert linear.method == "PCG"
    assert linear.tolerance == 1e-6
    assert linear.maxIterations == 1000

    assert nonlinear.method == "Newton"
    assert nonlinear.tolerance == 1e-8
    assert nonlinear.maxIterations == 50
