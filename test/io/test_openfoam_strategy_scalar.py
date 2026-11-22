# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The float reader in :data:`READ_DISPATCH` tolerates dimensioned scalars.

OpenFOAM writes a kinematic-viscosity entry as ``nu [ 0 2 -1 0 0 0 0 ] 1e-05``;
the strategy coerces float-typed config fields straight from that raw string, so
the reader must strip the dimensionSet block (and an optional leading keyword)
before converting. A plain number must still round-trip unchanged.
"""

import pytest

from neofoam.io.strategies.openfoam_strategy import _read_scalar


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("nu [ 0 2 -1 0 0 0 0 ] 1e-05", 1e-05),
        ("[ 0 2 -1 0 0 0 0 ] 1e-05", 1e-05),
        ("[ 0 2 -1 0 0 0 0 ] 1e-06", 1e-06),
        ("[ 0 2 -1 0 0 0 0 ] 0", 0.0),
        ("1e-05", 1e-05),
        ("42", 42.0),
    ],
)
def test_read_scalar_parses_dimensioned_and_plain(raw: str, expected: float) -> None:
    assert _read_scalar(raw) == expected
