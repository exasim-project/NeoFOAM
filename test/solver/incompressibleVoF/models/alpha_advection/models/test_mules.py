# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the MULES controls, ``alphaPhiUn`` and the alpha ddt off-centring.

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
  vary content through the format's reader/writer, never text-patch a copy),
  mirroring ``test_selection.py``'s technique.

The ``alphaPhiUn`` tests exercise ``../../../cases/vofRow4``'s executed
pipeline plus one live ``alpha_advection`` call, through
``_alpha_phi_un_worker.py`` (its own ``Foam::Time`` — see TEST_STYLE's "one
Foam::Time per process").

The sub-cycle tests do the same on ``../../../cases/vofRow4SubCycle``
(``nAlphaSubCycles 3``) through ``_sub_cycle_worker.py``, and assert on the
``Foam::Time`` state each sub-step is solved at — the thing
``alphaEqnSubCycle.H`` actually manipulates, and what every time-dependent
boundary condition reads.

The ``mixture.correct()`` test runs both cases through
``_mixture_correct_worker.py``, which logs ``Foam::Time``'s ``deltaT`` on each
call: inside the sub-cycle that is ``deltaT/nAlphaSubCycles``, after it the real
time step, so the log shows both how many corrections happen and on which side
of ``alphaEqnSubCycle.H`` each one falls.

The ``alphaApplyPrevCorr`` tests run ``../../../cases/vofRow4PrevCorr`` (which
is ``vofRow4`` plus that one entry) over three time steps through
``_prev_corr_worker.py``, with and without the flag, and assert on the
``talphaPhi1Corr0`` slot itself — whether it carries anything, and whether what
it carries is ``alphaPhi10 - talphaPhi1UD`` as ``alphaEqn.H:230`` defines it.
What the seeded correction then *does* to the solution is the
``alphaApplyPrevCorr_on`` entry of
``test/solver/incompressibleVoF/test_mules_regimes.py``, against native interFoam.

The off-centring is covered twice: ``alpha_ddt_off_centring``'s decision table
directly (it takes only the values a case would supply, so it needs no mesh),
and then ``read_alpha_ddt_off_centring`` on the real
``../../../cases/vofRow4CrankNicolson`` pair through
``_crank_nicolson_worker.py``, which is what pins the ``system/fvSchemes`` read
and the rejection to the code the solver actually calls. What the off-centring
then *does* to the solution is a different question, answered against native
interFoam in ``test/solver/incompressibleVoF/test_cranknicolson_alpha_ddt.py``.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pybFoam import dictionary

from neofoam.solver.incompressibleVoF.models.alpha_advection.models.mules import (
    alpha_ddt_off_centring,
    read_alpha_controls,
)

_VOF_ROW4 = Path(__file__).parents[3] / "cases" / "vofRow4"
_VOF_ROW4_SUB_CYCLE = Path(__file__).parents[3] / "cases" / "vofRow4SubCycle"
_VOF_ROW4_CRANK_NICOLSON = Path(__file__).parents[3] / "cases" / "vofRow4CrankNicolson"
_VOF_ROW4_CRANK_NICOLSON_SUB_CYCLE = (
    Path(__file__).parents[3] / "cases" / "vofRow4CrankNicolsonSubCycle"
)
_VOF_ROW4_PREV_CORR = Path(__file__).parents[3] / "cases" / "vofRow4PrevCorr"
_ALPHA_PHI_UN_WORKER = Path(__file__).parent / "_alpha_phi_un_worker.py"
_SUB_CYCLE_WORKER = Path(__file__).parent / "_sub_cycle_worker.py"
_MIXTURE_CORRECT_WORKER = Path(__file__).parent / "_mixture_correct_worker.py"
_PREV_CORR_WORKER = Path(__file__).parent / "_prev_corr_worker.py"
_CRANK_NICOLSON_WORKER = Path(__file__).parent / "_crank_nicolson_worker.py"

_CASES = Path(__file__).parent.parent / "cases"


def test_read_alpha_controls_reads_all_keys_from_the_real_damBreak_fvSolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The keys the real file sets -> its values, not the defaults.

    ``alphaApplyPrevCorr`` is not one of them: the damBreak tutorial leaves it
    out, so it comes back at alphaControls.H's ``false``.
    """
    monkeypatch.chdir(_CASES / "damBreak_mules")
    assert read_alpha_controls("alpha.water") == (2, 1, True, False)


def test_read_alpha_controls_falls_back_to_defaults_for_a_nonmatching_alpha_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``alpha.oil`` doesn't match the ``"alpha.water.*"`` regex key -> the
    'no solver dict' branch -> the documented defaults."""
    monkeypatch.chdir(_CASES / "damBreak_mules")
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


@pytest.fixture(scope="module")
def alpha_phi_un_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the VoF pipeline on ``cases/vofRow4`` and one ``alpha_advection`` call."""
    case = tmp_path_factory.mktemp("vofRow4_alphaPhiUn") / "case"
    shutil.copytree(_VOF_ROW4, case)
    subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [sys.executable, str(_ALPHA_PHI_UN_WORKER), str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    result: dict[str, object] = json.loads((case / "alpha_phi_un.json").read_text())
    return result


def test_alpha_phi_un_is_registered_by_build(
    alpha_phi_un_run: dict[str, object],
) -> None:
    assert alpha_phi_un_run["registered_name"] == "alphaPhiUn"


def test_alpha_phi_un_is_zero_before_the_alpha_solve(
    alpha_phi_un_run: dict[str, object],
) -> None:
    assert alpha_phi_un_run["before"] == [0.0, 0.0, 0.0]


def test_alpha_phi_un_is_updated_by_the_alpha_solve(
    alpha_phi_un_run: dict[str, object],
) -> None:
    # MULES fills the compressed flux in place; on vofRow4's real
    # rho1=1000/rho2=1/g=(0 -9.81 0) damBreak setup this is a fixed,
    # hand-verified (not just "changed") value.
    assert alpha_phi_un_run["after"] == pytest.approx(
        [0.6223526056240543, 1.5471535801518843, 1.7421947126197908]
    )


def test_alpha_phi_un_stays_the_same_registered_object_after_the_solve(
    alpha_phi_un_run: dict[str, object],
) -> None:
    # Proves the fix: a fresh same-named field would have self-deregistered
    # from the objectRegistry (the ``failed lookup of alphaPhiUn`` bug); the
    # solve must assign into the persistent one instead.
    assert alpha_phi_un_run["still_registered_name"] == "alphaPhiUn"


# --- alphaEqnSubCycle.H: the sub-cycle runs on a real Foam::Time -----------
#
# ``cases/vofRow4SubCycle`` is ``cases/vofRow4`` with ``nAlphaSubCycles 3``
# (deltaT 0.25, adjustTimeStep off), so the expected sub-times are exact
# eighths-free thirds of one step and can be written down.


@pytest.fixture(scope="module")
def sub_cycle_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """One ``alpha_advection`` call on a 3-sub-cycle case; the Time state it saw."""
    case = tmp_path_factory.mktemp("vofRow4SubCycle") / "case"
    shutil.copytree(_VOF_ROW4_SUB_CYCLE, case)
    subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [sys.executable, str(_SUB_CYCLE_WORKER), str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    result: dict[str, object] = json.loads((case / "sub_cycle.json").read_text())
    return result


def test_the_sub_cycle_advances_time_once_per_sub_step(
    sub_cycle_run: dict[str, object],
) -> None:
    # The defect this replaces scaled the flux instead and left Time frozen, so
    # every sub-step saw the same end-of-step time — and a time-driven patch
    # field fired once per step instead of once per sub-step.
    times = [step["time"] for step in sub_cycle_run["sub_steps"]]  # type: ignore[index]
    assert times == pytest.approx([0.25 / 3, 0.5 / 3, 0.25])


def test_the_sub_cycle_divides_the_time_step(sub_cycle_run: dict[str, object]) -> None:
    deltas = [step["deltaT"] for step in sub_cycle_run["sub_steps"]]  # type: ignore[index]
    assert deltas == pytest.approx([0.25 / 3] * 3)


def test_each_sub_step_gets_its_own_time_index(sub_cycle_run: dict[str, object]) -> None:
    # A fresh time index per sub-step is what makes fvPatchField::updateCoeffs()
    # re-evaluate — Foam::Time::subCycle scales the index by nSubCycles first.
    indices = [step["timeIndex"] for step in sub_cycle_run["sub_steps"]]  # type: ignore[index]
    assert indices == [1, 2, 3]


def test_the_sub_cycle_restores_the_time_state(sub_cycle_run: dict[str, object]) -> None:
    # endSubCycle() puts back time, deltaT and index, so the momentum/pressure
    # stage after the alpha solve runs on the real time step.
    assert sub_cycle_run["after"] == sub_cycle_run["before"]


# --- interFoam.C:154: one more mixture.correct() after the alpha block -----


@pytest.mark.parametrize(
    "source_case, expected_delta_t_at_each_correct",
    [
        # Both cases run nAlphaCorr 2 with MULESCorr on, so one alpha_eqn pass
        # corrects the mixture three times (the implicit-upwind predictor plus
        # the two correctors); the trailing entry at the *undivided* deltaT is
        # interFoam.C:154, issued once for the whole alpha block.
        pytest.param(_VOF_ROW4, [0.25] * 3 + [0.25], id="nAlphaSubCycles_1"),
        pytest.param(_VOF_ROW4_SUB_CYCLE, [0.25 / 3] * 9 + [0.25], id="nAlphaSubCycles_3"),
    ],
)
def test_the_mixture_is_corrected_once_more_after_the_alpha_block(
    tmp_path: Path,
    source_case: Path,
    expected_delta_t_at_each_correct: list[float],
) -> None:
    """interFoam calls ``mixture.correct()`` once after ``alphaEqnSubCycle.H``.

    Idempotent for these two cases — ``interfaceProperties::correct()`` is a pure
    function of alpha1 without an ``alphaContactAngle`` patch, which is why the
    omission was invisible in the field values — so the call is counted rather
    than measured. It is not once *per* sub-cycle: nine inner corrections at
    ``deltaT/3``, then a single one back on the real time step.
    """
    case = tmp_path / "case"
    shutil.copytree(source_case, case)
    subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [sys.executable, str(_MIXTURE_CORRECT_WORKER), str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    run = json.loads((case / "mixture_correct.json").read_text())

    assert run["delta_t_at_each_correct"] == pytest.approx(expected_delta_t_at_each_correct)


# --- alphaEqn.H:133-150 and 228-236: the alphaApplyPrevCorr flux cache -----


@pytest.fixture(scope="module")
def prev_corr_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict[str, object]]]:
    """``cases/vofRow4PrevCorr`` with and without the flag, three time steps each.

    The ``off`` run is the same case with ``alphaApplyPrevCorr`` removed through
    pybFoam's own dictionary writer, so the pair provably differs in that one
    entry and nothing else.
    """
    runs: dict[str, list[dict[str, object]]] = {}
    for label in ("off", "on"):
        case = tmp_path_factory.mktemp(f"vofRow4PrevCorr_{label}") / "case"
        shutil.copytree(_VOF_ROW4_PREV_CORR, case)
        if label == "off":
            fv_solution = dictionary.read(str(case / "system" / "fvSolution"))
            fv_solution.subDict("solvers").subDict("alpha.water").remove("alphaApplyPrevCorr")
            fv_solution.write(str(case / "system" / "fvSolution"))
        subprocess.run(
            ["blockMesh", "-case", str(case)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        subprocess.run(
            [sys.executable, str(_PREV_CORR_WORKER), str(case)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        runs[label] = json.loads((case / "prev_corr.json").read_text())["passes"]
    return runs


def test_the_compression_flux_cache_stays_empty_without_alphaApplyPrevCorr(
    prev_corr_runs: dict[str, list[dict[str, object]]],
) -> None:
    # alphaEqn.H:233-236 — the ``else`` branch clears the cache on every pass, so
    # a case that never asks for it can never be seeded by it.
    assert [step["cache_after"] for step in prev_corr_runs["off"]] == [None, None, None]


def test_the_compression_flux_cache_is_seeded_only_from_the_second_pass(
    prev_corr_runs: dict[str, list[dict[str, object]]],
) -> None:
    # alphaEqn.H:133 — ``alphaApplyPrevCorr && talphaPhi1Corr0.valid()``: there is
    # nothing to apply until a pass has filled the slot, and what each pass
    # applies is exactly what the pass before it left there.
    passes = prev_corr_runs["on"]
    assert passes[0]["cache_before"] is None
    assert all(step["cache_after"] is not None for step in passes)
    assert [step["cache_before"] for step in passes[1:]] == [
        step["cache_after"] for step in passes[:-1]
    ]


def test_the_cached_flux_is_the_correction_the_pass_applied(
    prev_corr_runs: dict[str, list[dict[str, object]]],
) -> None:
    """alphaEqn.H:230 — ``talphaPhi1Corr0 = alphaPhi10 - talphaPhi1Corr0``.

    Checked as the identity native writes it (against the pass's own upwind
    predictor flux) rather than against recorded numbers, so it pins the meaning
    of the cache and not just its value.
    """
    for index, step in enumerate(prev_corr_runs["on"]):
        expected = [
            alpha_phi10 - upwind
            for alpha_phi10, upwind in zip(step["alphaPhi10"], step["upwind"])  # type: ignore[call-overload]
        ]
        assert step["cache_after"] == pytest.approx(expected), f"pass {index}"


def test_the_cached_correction_changes_the_flux_once_it_is_non_zero(
    prev_corr_runs: dict[str, list[dict[str, object]]],
) -> None:
    """The behavioural statement of the whole cache, on this case's own timeline.

    Passes 1 and 2 must agree with the flag off: pass 1 has nothing cached, and
    what it caches is exactly zero (this case's Courant number limits the whole
    compression correction away on the first step). Pass 3 is the first one
    seeded with a non-zero correction, and it moves the flux the momentum
    equation is then fed.
    """
    off, on = prev_corr_runs["off"], prev_corr_runs["on"]
    assert on[0]["cache_after"] == [0.0, 0.0, 0.0]
    assert [step["alphaPhi10"] for step in on[:2]] == [step["alphaPhi10"] for step in off[:2]]
    assert on[2]["alphaPhi10"] != off[2]["alphaPhi10"]


# --- alphaEqn.H: the ddt(alpha) off-centring coefficient ------------------


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


def test_alpha_ddt_off_centring_rejects_sub_cycling_with_crank_nicolson() -> None:
    """The combination interFoam calls a FatalError, not something to approximate."""
    with pytest.raises(ValueError, match="Sub-cycling is not supported"):
        alpha_ddt_off_centring("CrankNicolson", 0.5, 2, True)


def test_alpha_ddt_off_centring_rejects_an_unsupported_ddt_scheme() -> None:
    """``backward`` parses and runs elsewhere, but MULES has no branch for it."""
    with pytest.raises(ValueError, match="only Euler and CrankNicolson"):
        alpha_ddt_off_centring("backward", 0.0, 1, True)


@pytest.fixture(scope="module")
def crank_nicolson_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Two time steps of ``cases/vofRow4CrankNicolson``; the ocCoeff each one read."""
    case = tmp_path_factory.mktemp("vofRow4CrankNicolson") / "case"
    shutil.copytree(_VOF_ROW4_CRANK_NICOLSON, case)
    subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [sys.executable, str(_CRANK_NICOLSON_WORKER), str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    result: dict[str, object] = json.loads((case / "crank_nicolson.json").read_text())
    return result


def test_the_first_time_step_of_a_crank_nicolson_run_is_off_centred_as_euler(
    crank_nicolson_run: dict[str, object],
) -> None:
    # ``CrankNicolson 0.5`` in the case's fvSchemes, but the alpha flux it would
    # off-centre against does not exist until a step has been taken.
    assert crank_nicolson_run["oc_coeffs"] == [0.0, 0.5]  # type: ignore[comparison-overlap]


def test_a_crank_nicolson_run_does_not_reject_a_single_alpha_sub_cycle(
    crank_nicolson_run: dict[str, object],
) -> None:
    assert crank_nicolson_run["error"] is None


def test_alpha_advection_rejects_sub_cycling_a_crank_nicolson_case(
    tmp_path: Path,
) -> None:
    """The rejection reaches the solve, not just the helper it is written in.

    ``cases/vofRow4CrankNicolsonSubCycle`` is the same case with
    ``nAlphaSubCycles 2``; the very first alpha solve must refuse it rather than
    silently integrate the sub-steps as Euler (which is what makes the
    combination look like it works).
    """
    case = tmp_path / "case"
    shutil.copytree(_VOF_ROW4_CRANK_NICOLSON_SUB_CYCLE, case)
    subprocess.run(
        ["blockMesh", "-case", str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [sys.executable, str(_CRANK_NICOLSON_WORKER), str(case)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    run = json.loads((case / "crank_nicolson.json").read_text())

    assert run["oc_coeffs"] == []
    assert "Sub-cycling is not supported" in run["error"]
