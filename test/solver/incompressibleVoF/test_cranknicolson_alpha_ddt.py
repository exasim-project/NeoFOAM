# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The alpha equation under a Crank-Nicolson ``ddt``, against native interFoam.

``alphaEqn.H`` reads ``ddtSchemes/ddt(alpha)`` and, for ``CrankNicolson``,
changes three things at once: it transports with an off-centred flux
``phiCN = cnCoeff*phi + (1 - cnCoeff)*phi.oldTime()``, it converts the resulting
``alphaPhi10`` back to an end-of-time-step flux against ``alphaPhi10.oldTime()``,
and it forms ``rhoPhi`` from that converted flux and the *raw* ``phi``. Which of
those fire depends on the case, so each ``parametrize`` entry below turns on a
different subset (TEST_STYLE rule 10):

* ``Euler_3_outer_correctors`` — the tutorial's own ``ddtSchemes``, so ocCoeff
  is 0 and none of it fires. The guard for the claim that the Crank-Nicolson
  code path leaves the Euler one alone, including the part easiest to break:
  ``phi``'s old-time slot must not be read at all, because reading it rolls the
  slot at a point native does not.
* ``CrankNicolson`` — ``default CrankNicolson 0.5`` at the tutorial's single
  outer corrector. ``phiCN`` is still ``phi`` here (nothing has written ``phi``
  yet when the alpha equation runs, so its old time equals it), which isolates
  the ``alphaPhi10`` conversion and the ``rhoPhi`` branch.
* ``CrankNicolson_3_outer_correctors`` — the same with ``nOuterCorrectors 3``.
  From the second outer corrector ``phi`` is the end-of-step estimate while its
  old time is the start-of-step value, so ``phiCN`` genuinely differs from
  ``phi`` and the transport itself is off-centred.
* ``alpha_Euler_momentum_CrankNicolson`` — ``ddt(alpha) Euler`` under a
  ``CrankNicolson`` default. ocCoeff is 0 but ``ddt(rho,U)`` is not Euler, so
  native takes its *second* ``rhoPhi`` expression with the conversion switched
  off; the two expressions must agree exactly there.

The ``fvSchemes`` variants are checked-in files under ``cases/ddtSchemes/`` that
replace the tutorial's, because a scheme spec (``CrankNicolson 0.5``) is two
tokens and the dictionary writer can only set an entry from a single value.

**Oracle and horizon** are ``test_mules_regimes``': the same ``tutorials/damBreak``
run to 0.05 s (~7 adaptive steps) by both solvers reading the same dictionaries,
compared at ``rtol = atol = 1e-10``. Both runs go through subprocesses — a
process may construct exactly one ``Foam::Time`` and this module runs four
configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

import pytest

from ..incompressibleFluid.comparison_helpers import compare_solver_fields
from .comparison_helpers import FIELDS_TO_COMPARE, run_dambreak_regime

PimpleControls = Mapping[str, Union[bool, float]]

_DDT_SCHEMES = Path(__file__).parent / "cases" / "ddtSchemes"

SCHEMES = [
    pytest.param((None, {"nOuterCorrectors": 3.0}), id="Euler_3_outer_correctors"),
    pytest.param((_DDT_SCHEMES / "crankNicolson" / "fvSchemes", {}), id="CrankNicolson"),
    pytest.param(
        (_DDT_SCHEMES / "crankNicolson" / "fvSchemes", {"nOuterCorrectors": 3.0}),
        id="CrankNicolson_3_outer_correctors",
    ),
    pytest.param(
        (_DDT_SCHEMES / "alphaEulerMomentumCrankNicolson" / "fvSchemes", {}),
        id="alpha_Euler_momentum_CrankNicolson",
    ),
]


@dataclass
class SchemeRun:
    """One ddt configuration, run to ``endTime`` by both solvers."""

    fv_schemes: Optional[Path]
    pimple_controls: PimpleControls
    python_case: Path
    native_case: Path


@pytest.fixture(scope="module", params=SCHEMES)
def scheme_run(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> SchemeRun:
    """Run one ddt configuration with both solvers, once."""
    fv_schemes, pimple_controls = request.param
    root = tmp_path_factory.mktemp("crankNicolsonDdt")
    python_case = root / "incompressibleVoF"
    native_case = root / "interFoam"
    run_dambreak_regime(
        {},
        python_case,
        native_case,
        fv_schemes=fv_schemes,
        pimple_controls=pimple_controls,
    )
    return SchemeRun(fv_schemes, pimple_controls, python_case, native_case)


def test_alpha_ddt_scheme_matches_native_interFoam(scheme_run: SchemeRun) -> None:
    """The Python alpha equation reproduces interFoam under this ddt scheme."""
    all_match, failed_fields, failed_details = compare_solver_fields(
        scheme_run.python_case,
        scheme_run.native_case,
        FIELDS_TO_COMPARE,
        rtol=1e-10,
        atol=1e-10,
    )

    assert all_match, (
        f"fvSchemes {scheme_run.fv_schemes}, PIMPLE {dict(scheme_run.pimple_controls)}: "
        + ", ".join(
            f"{name}(abs={failed_details[name][0]:.3e}, rel={failed_details[name][1]:.3e})"
            for name in failed_fields
        )
    )
