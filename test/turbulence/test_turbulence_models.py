# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Tests for turbulence models.

NOTE: This file has been split into multiple focused test files:
- test_plugin_system.py: Plugin registration tests
- test_model_creation.py: Model instantiation tests
- test_model_setup.py: Setup and initialization tests
- test_file_reading.py: File parsing tests
- test_operations.py: Operations collection tests
- test_integration.py: Integration tests with real OpenFOAM cases

This file is kept for backward compatibility but may be removed in the future.
Use the individual test files for new tests.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - framework refactored (@Model.build decorator removed) - Old API imports removed"
)


def test_placeholder():
    """Placeholder test to prevent collection errors."""
    pass
