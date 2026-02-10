# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Shared fixtures for unit tests."""

import pytest
from pathlib import Path
import shutil


@pytest.fixture
def io_fixtures():
    """IO decorator test configs directory."""
    return Path(__file__).parent / "configs"


@pytest.fixture
def temp_fixture_copy(io_fixtures, tmp_path):
    """Copy a config file to temp dir for modification tests.

    Usage:
        test_file = temp_fixture_copy("shared.json")
        # Modify test_file without affecting original
    """

    def _copy(fixture_name):
        src = io_fixtures / fixture_name
        dst = tmp_path / fixture_name
        shutil.copy(src, dst)
        return dst

    return _copy
