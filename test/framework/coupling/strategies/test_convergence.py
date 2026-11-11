# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The convergence measures of a coupling, on hand-checked changes."""

from typing import Any

import numpy as np
import pytest

from neofoam.framework.coupling import (
    RelativeConvergence,
    SideChange,
)


@pytest.mark.parametrize(
    ("current", "converged"),
    [(100.0 + 1e-4, True), (100.0 + 1e-2, False)],
    ids=["change 1e-6 relative", "change 1e-4 relative"],
)
def test_relative_convergence_compares_the_change_with_the_limit(
    current: float, converged: bool
) -> None:
    change = SideChange({"T_sink": np.array([100.0])}, {"T_sink": np.array([current])})

    assert RelativeConvergence(data="T_sink", limit=1e-5)([change]) is converged


def test_relative_convergence_ignores_other_fields() -> None:
    changes: list[Any] = [SideChange({"u": np.array([1.0])}, {"u": np.array([2.0])})]

    assert RelativeConvergence(data="T", limit=1e-5)(changes) is True
