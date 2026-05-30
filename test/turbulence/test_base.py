# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the ``TurbulenceModel`` Protocol surface (cross-model).

Each native model is built the way the solver builds it (from the case), then
checked for protocol conformance — no hand-injected constructor args.
"""

from typing import Any

import pytest

from neofoam.turbulence.base import TurbulenceModel

from turbulence.conftest import CASES, Case, build_as_solver

NATIVE_CASES = [c for c in CASES if c.selection["resolves_to"] == "native"]


@pytest.mark.parametrize("case", NATIVE_CASES, ids=lambda c: c.name)
def test_model_built_as_solver_satisfies_protocol(case: Case) -> None:
    assert isinstance(build_as_solver(case), TurbulenceModel)


def test_incomplete_class_is_not_a_turbulence_model() -> None:
    class Incomplete:
        def nut(self) -> Any:
            return 0.0

        def nu(self) -> Any:
            return 0.0

        # missing divDevReff and correct

    assert not isinstance(Incomplete(), TurbulenceModel)
