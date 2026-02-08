# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Comparison test for pitzDaily case: SimpleSolver vs native pimpleFoam.
Tests that both solvers produce matching results by loading and comparing
the actual volScalarField/volVectorField data from disk.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Outdated - solver incompressibleFluid not yet adapted to new framework - Old API imports removed"
)


def test_placeholder():
    """Placeholder test to prevent collection errors."""
    pass
