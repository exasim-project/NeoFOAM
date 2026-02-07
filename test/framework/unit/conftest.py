# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Shared fixtures for unit tests."""

import pytest
from pathlib import Path
import shutil


@pytest.fixture(scope="session")
def fixtures_root():
    """Root directory for all test fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def io_fixtures(fixtures_root):
    """IO decorator test fixtures directory."""
    return fixtures_root / "io_decorator"


@pytest.fixture
def temp_fixture_copy(io_fixtures, tmp_path):
    """Copy a fixture to temp dir for modification tests.
    
    Usage:
        test_file = temp_fixture_copy("fvSolution.yaml")
        # Modify test_file without affecting fixture
    """
    def _copy(fixture_name):
        src = io_fixtures / fixture_name
        dst = tmp_path / fixture_name
        shutil.copy(src, dst)
        return dst
    return _copy
