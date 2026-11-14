# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""advectorVoF's time-dependent flux factor, read from the case's ``isoAdvector`` dict.

The expected factors are cosines of 15, 75 and 105 degrees (``spirallingFlow 3``
at ``t + dt/2`` = 0.25, 1.25 and 1.75), negated once ``t`` reaches
``reverseTime 1``, and exactly once. They are exact to a few ulps, so ``atol = 1e-15``.
"""

from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from neofoam.algorithms.vof.advection import _FluxModulation

CASES = Path(__file__).parent / "cases"


def test_flux_modulation_spiralling_reversed_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each step's ``phi/phi0`` is the mid-step cosine, sign-flipped from ``reverseTime`` on."""
    monkeypatch.chdir(CASES / "spirallingReversedFlow")
    modulation = _FluxModulation.read()
    assert modulation is not None

    factors = [modulation.factor(t, dt) for t, dt in [(0.0, 0.5), (1.0, 0.5), (1.25, 1.0)]]

    assert_allclose(
        factors,
        [0.96592582628906829, -0.25881904510252076, 0.25881904510252076],
        rtol=0,
        atol=1e-15,
        err_msg="spirallingReversedFlow",
    )


def test_flux_modulation_absent_without_isoAdvector_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A case without the dict keeps its flux constant."""
    monkeypatch.chdir(CASES / "discUniformFlow")

    assert _FluxModulation.read() is None
