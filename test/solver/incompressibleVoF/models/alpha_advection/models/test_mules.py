# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for ``read_alpha_controls`` and ``alphaPhiUn`` (MULES: ``models/mules.py``).

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
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pybFoam import dictionary

from neofoam.solver.incompressibleVoF.models.alpha_advection.models.mules import (
    read_alpha_controls,
)

_VOF_ROW4 = Path(__file__).parents[3] / "cases" / "vofRow4"
_ALPHA_PHI_UN_WORKER = Path(__file__).parent / "_alpha_phi_un_worker.py"

_CASES = Path(__file__).parent.parent / "cases"


def test_read_alpha_controls_reads_all_keys_from_the_real_damBreak_fvSolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three keys present -> the real file's values, not the defaults."""
    monkeypatch.chdir(_CASES / "damBreak_mules")
    assert read_alpha_controls("alpha.water") == (2, 1, True)


def test_read_alpha_controls_falls_back_to_defaults_for_a_nonmatching_alpha_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``alpha.oil`` doesn't match the ``"alpha.water.*"`` regex key -> the
    'no solver dict' branch -> the documented defaults."""
    monkeypatch.chdir(_CASES / "damBreak_mules")
    assert read_alpha_controls("alpha.oil") == (1, 1, False)


@pytest.mark.parametrize(
    "keys_to_remove, expected",
    [
        pytest.param(["nAlphaCorr"], (1, 1, True), id="nAlphaCorr_absent"),
        pytest.param(["MULESCorr"], (2, 1, False), id="MULESCorr_absent"),
        pytest.param(
            ["nAlphaCorr", "nAlphaSubCycles", "MULESCorr"],
            (1, 1, False),
            id="all_keys_absent",
        ),
    ],
)
def test_read_alpha_controls_falls_back_to_defaults_for_missing_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keys_to_remove: list[str],
    expected: tuple[int, int, bool],
) -> None:
    """Per-key defaulting: each key removed from a copy of the real
    ``"alpha.water.*"`` entry (via pybFoam's own reader/writer) falls back to
    its documented default while the remaining, still-present keys keep the
    file's values."""
    shutil.copytree(_CASES / "damBreak_mules", tmp_path, dirs_exist_ok=True)
    fv_solution_path = tmp_path / "system" / "fvSolution"

    fv_solution = dictionary.read(str(fv_solution_path))
    alpha_dict = fv_solution.subDict("solvers").subDict("alpha.water")
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
