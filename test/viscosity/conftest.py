# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared fixtures for viscosity tests.

``register_with`` mutates the global ``PluginSystem`` registry as an import
side-effect, and that state persists across a test session. The
``clean_viscosity_registry`` fixture snapshots the registry and removes any
spec a test registers, so registrations do not leak between tests.
"""

import subprocess
from pathlib import Path
from typing import Iterator

import pytest

from neofoam.core.plugin_system import PluginSystem

# Ensure the bundled native models (Newtonian) are registered before any test.
import neofoam.viscosity  # noqa: F401

#: Self-contained OpenFOAM cases shipped with the viscosity tests. Each holds a
#: real ``constant/transportProperties`` dictionary (no dict content is encoded
#: in the test modules).
CASES_DIR = Path(__file__).resolve().parent / "cases"
NEWTONIAN_CASE = CASES_DIR / "newtonian"  # transportModel Newtonian
CROSS_POWER_LAW_CASE = CASES_DIR / "crossPowerLaw"  # transportModel CrossPowerLaw


def _require_case(case_dir: Path) -> Path:
    if not (case_dir / "constant" / "transportProperties").is_file():
        pytest.skip(f"case not available: {case_dir}")
    return case_dir


@pytest.fixture
def newtonian_case() -> Path:
    """Newtonian transportProperties case (copied from ``tutorials/pitzDaily``)."""
    return _require_case(NEWTONIAN_CASE)


@pytest.fixture
def cross_power_law_case() -> Path:
    """Non-Newtonian (CrossPowerLaw) transportProperties case."""
    return _require_case(CROSS_POWER_LAW_CASE)


def _check_openfoam_available() -> bool:
    """Return True iff ``blockMesh`` is on PATH and runnable."""
    try:
        result = subprocess.run(["blockMesh", "-help"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


#: Skip marker for tests that read OpenFOAM dictionaries via pybFoam.
requires_openfoam = pytest.mark.skipif(
    not _check_openfoam_available(),
    reason="OpenFOAM not available",
)


@pytest.fixture
def clean_viscosity_registry() -> Iterator[None]:
    """Remove any viscosityModel plugins registered during the test."""
    registry = PluginSystem.get_registered("viscosityModel")
    before = list(registry.plugin_registry) if registry else []
    try:
        yield
    finally:
        registry = PluginSystem.get_registered("viscosityModel")
        if registry is not None:
            for plugin_cls in list(registry.plugin_registry):
                if plugin_cls not in before:
                    PluginSystem.remove_plugin_model("viscosityModel", plugin_cls)
