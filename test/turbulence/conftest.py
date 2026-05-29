# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared fixtures for turbulence tests.

``register_with`` mutates the global ``PluginSystem`` registry as an import
side-effect, and that state persists across a test session. The
``clean_turbulence_registry`` fixture snapshots the registry and removes any
spec a test registers (e.g. a throwaway ``kEpsilon`` native), so registrations
do not leak between tests.
"""

import subprocess
from pathlib import Path
from typing import Iterator

import pytest

from neofoam.core.plugin_system import PluginSystem

# Ensure the bundled native models (laminar) are registered before any test.
import neofoam.turbulence  # noqa: F401

#: Self-contained OpenFOAM cases shipped with the turbulence tests. Each holds a
#: real ``constant/turbulenceProperties`` dictionary (no dict content is encoded
#: in the test modules).
CASES_DIR = Path(__file__).resolve().parent / "cases"
RAS_CASE = CASES_DIR / "ras_kEpsilon"  # simulationType RAS, RASModel kEpsilon
LES_CASE = CASES_DIR / "les_Smagorinsky"  # simulationType LES, LESModel Smagorinsky
LAMINAR_CASE = CASES_DIR / "laminar"  # simulationType laminar


def _require_case(case_dir: Path) -> Path:
    if not (case_dir / "constant" / "turbulenceProperties").is_file():
        pytest.skip(f"case not available: {case_dir}")
    return case_dir


@pytest.fixture
def ras_case() -> Path:
    """RAS / kEpsilon case (copied verbatim from ``tutorials/hotRoom``)."""
    return _require_case(RAS_CASE)


@pytest.fixture
def les_case() -> Path:
    """LES / Smagorinsky case."""
    return _require_case(LES_CASE)


@pytest.fixture
def laminar_case() -> Path:
    """Laminar case (no RAS/LES sub-dictionary)."""
    return _require_case(LAMINAR_CASE)


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
def clean_turbulence_registry() -> Iterator[None]:
    """Remove any turbulenceModel plugins registered during the test."""
    registry = PluginSystem.get_registered("turbulenceModel")
    before = list(registry.plugin_registry) if registry else []
    try:
        yield
    finally:
        registry = PluginSystem.get_registered("turbulenceModel")
        if registry is not None:
            for plugin_cls in list(registry.plugin_registry):
                if plugin_cls not in before:
                    PluginSystem.remove_plugin_model("turbulenceModel", plugin_cls)
