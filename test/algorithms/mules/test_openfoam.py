# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the MULES controls and the alpha ddt off-centring decision table.

``read_alpha_controls`` re-reads ``system/fvSolution`` from the cwd on every
call and resolves ``alpha_name`` against the ``solvers`` sub-dict keys using
OpenFOAM's own regex dict-key matching (``found``/``subDict`` default to
``keyType::REGEX`` — see ``dictionary.H``), so the real damBreak
``"alpha.water.*"`` entry is found for ``alpha_name="alpha.water"``.

- ``cases/damBreak_mules/system/fvSolution`` is the real interFoam damBreak
  fvSolution (``nAlphaCorr 2; nAlphaSubCycles 1; MULESCorr yes;``) — used
  as-is for the all-keys-present case and to prove the regex resolution.
- The keys-absent cases are derived from that same real file at test time via
  pybFoam's own ``dictionary.read``/``remove``/``write`` (TEST_STYLE rule 3:
  vary content through the format's reader/writer, never text-patch a copy).

``alpha_ddt_off_centring`` takes only the values a case would supply, so its
decision table needs no mesh. The live ``system/fvSchemes`` read is covered in
``test/solver/incompressibleVoF/models/alpha_advection/models/test_mules.py``.
"""

import shutil
from pathlib import Path

import pytest
from pybFoam import dictionary

from neofoam.algorithms.mules import alpha_ddt_off_centring, read_alpha_controls

_CASES = Path(__file__).parent / "cases"


def test_read_alpha_controls_resolves_the_regex_key_and_defaults_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The keys the real file sets -> its values, not the defaults.

    ``alphaApplyPrevCorr`` is not one of them: the damBreak tutorial leaves it
    out, so it comes back at alphaControls.H's ``false``. ``alpha.oil`` doesn't
    match the ``"alpha.water.*"`` regex key at all -> the 'no solver dict'
    branch -> the documented defaults.
    """
    monkeypatch.chdir(_CASES / "damBreak_mules")
    assert read_alpha_controls("alpha.water") == (2, 1, True, False)
    assert read_alpha_controls("alpha.oil") == (1, 1, False, False)


@pytest.mark.parametrize(
    "entries_to_set, keys_to_remove, expected",
    [
        pytest.param({}, ["nAlphaCorr"], (1, 1, True, False), id="nAlphaCorr_absent"),
        pytest.param({}, ["MULESCorr"], (2, 1, False, False), id="MULESCorr_absent"),
        pytest.param(
            {},
            ["nAlphaCorr", "nAlphaSubCycles", "MULESCorr"],
            (1, 1, False, False),
            id="all_keys_absent",
        ),
        # The only key the tutorial never sets, so it is read rather than
        # defaulted only when a case adds it (floatingObject, DTCHull*).
        pytest.param(
            {"alphaApplyPrevCorr": True}, [], (2, 1, True, True), id="alphaApplyPrevCorr_present"
        ),
    ],
)
def test_read_alpha_controls_falls_back_to_defaults_for_missing_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entries_to_set: dict[str, bool],
    keys_to_remove: list[str],
    expected: tuple[int, int, bool, bool],
) -> None:
    """Per-key defaulting: each key removed from a copy of the real
    ``"alpha.water.*"`` entry (via pybFoam's own reader/writer) falls back to
    its documented default while the remaining, still-present keys keep the
    file's values."""
    shutil.copytree(_CASES / "damBreak_mules", tmp_path, dirs_exist_ok=True)
    fv_solution_path = tmp_path / "system" / "fvSolution"

    fv_solution = dictionary.read(str(fv_solution_path))
    alpha_dict = fv_solution.subDict("solvers").subDict("alpha.water")
    for key, value in entries_to_set.items():
        alpha_dict.set(key, value)
    for key in keys_to_remove:
        alpha_dict.remove(key)
    fv_solution.write(str(fv_solution_path))

    monkeypatch.chdir(tmp_path)
    assert read_alpha_controls("alpha.water") == expected


@pytest.mark.parametrize(
    "scheme_name, scheme_coefficient, n_alpha_sub_cycles, after_first_time_step, expected",
    [
        pytest.param("Euler", 0.0, 1, True, 0.0, id="Euler"),
        pytest.param("localEuler", 0.0, 1, True, 0.0, id="localEuler"),
        # An Euler ddt(alpha) is never off-centred, so sub-cycling it is fine.
        pytest.param("Euler", 0.0, 4, True, 0.0, id="Euler_sub_cycled"),
        pytest.param("CrankNicolson", 0.5, 1, True, 0.5, id="CrankNicolson"),
        pytest.param("CrankNicolson", 0.9, 1, True, 0.9, id="CrankNicolson_0_9"),
        # No old-time alpha flux to off-centre against yet.
        pytest.param("CrankNicolson", 0.5, 1, False, 0.0, id="CrankNicolson_first_step"),
    ],
)
def test_alpha_ddt_off_centring_of_a_supported_scheme(
    scheme_name: str,
    scheme_coefficient: float,
    n_alpha_sub_cycles: int,
    after_first_time_step: bool,
    expected: float,
) -> None:
    """ocCoeff is the scheme's coefficient only once Crank-Nicolson has started."""
    assert (
        alpha_ddt_off_centring(
            scheme_name, scheme_coefficient, n_alpha_sub_cycles, after_first_time_step
        )
        == expected
    )


@pytest.mark.parametrize(
    "arguments, message",
    [
        # The combination interFoam calls a FatalError, not something to approximate.
        pytest.param(
            ("CrankNicolson", 0.5, 2, True), "Sub-cycling is not supported", id="sub_cycled_CN"
        ),
        # ``backward`` parses and runs elsewhere, but MULES has no branch for it.
        pytest.param(
            ("backward", 0.0, 1, True), "only Euler and CrankNicolson", id="unsupported_scheme"
        ),
    ],
)
def test_alpha_ddt_off_centring_rejects_what_mules_has_no_branch_for(
    arguments: tuple[str, float, int, bool], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        alpha_ddt_off_centring(*arguments)
