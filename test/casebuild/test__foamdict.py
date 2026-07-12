# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Unit tests for the low-level OpenFOAM dict patcher (``apply_overrides``).

Exercises the value dispatch and dotted-key walk directly on a staged copy of a
committed ``controlDict``, re-reading through pybFoam to confirm each write — no
pipeline wrapper involved. The checked-in dict is never mutated (staged into
``tmp_path`` first).
"""

import shutil
from pathlib import Path

import pybFoam as pyf
import pytest

from neofoam.casebuild._foamdict import apply_overrides, remove_entries

CONTROLDICT = Path(__file__).parent / "cases" / "cavity" / "system" / "controlDict"


def _staged(tmp_path: Path) -> Path:
    dst = tmp_path / "controlDict"
    shutil.copy(CONTROLDICT, dst)
    return dst


def test_apply_overrides_sets_existing_scalar(tmp_path: Path) -> None:
    cd = _staged(tmp_path)
    apply_overrides(cd, {"endTime": 0.03})
    assert pyf.dictionary.read(str(cd)).get_scalar("endTime") == pytest.approx(0.03)


def test_apply_overrides_adds_missing_key(tmp_path: Path) -> None:
    cd = _staged(tmp_path)
    assert not pyf.dictionary.read(str(cd)).found("maxCo")
    apply_overrides(cd, {"maxCo": 0.2})
    assert pyf.dictionary.read(str(cd)).get_scalar("maxCo") == pytest.approx(0.2)


def test_apply_overrides_bool_writes_openfoam_switch(tmp_path: Path) -> None:
    cd = _staged(tmp_path)
    apply_overrides(cd, {"adjustTimeStep": True})
    assert str(pyf.dictionary.read(str(cd)).get_word("adjustTimeStep")) == "yes"


def test_apply_overrides_dotted_key_targets_subdict(tmp_path: Path) -> None:
    cd = _staged(tmp_path)
    apply_overrides(cd, {"PIMPLE.nCorrectors": 2})
    sub = pyf.dictionary.read(str(cd)).subDict("PIMPLE")
    assert sub.get_scalar("nCorrectors") == pytest.approx(2)


def test_apply_overrides_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        apply_overrides(tmp_path / "nope", {"endTime": 0.1})


def test_apply_overrides_rejects_unsupported_type(tmp_path: Path) -> None:
    cd = _staged(tmp_path)
    with pytest.raises(TypeError):
        apply_overrides(cd, {"foo": [1, 2, 3]})


def test_remove_entries_drops_the_key(tmp_path: Path) -> None:
    cd = _staged(tmp_path)
    assert pyf.dictionary.read(str(cd)).found("deltaT")
    remove_entries(cd, ["deltaT"])
    assert not pyf.dictionary.read(str(cd)).found("deltaT")


def test_remove_entries_leaves_other_keys_intact(tmp_path: Path) -> None:
    cd = _staged(tmp_path)
    end_time = pyf.dictionary.read(str(cd)).get_scalar("endTime")
    remove_entries(cd, ["deltaT"])
    # a same-prefix neighbour must survive the word-boundary match
    assert pyf.dictionary.read(str(cd)).get_scalar("endTime") == pytest.approx(end_time)


def test_remove_entries_is_idempotent_for_absent_key(tmp_path: Path) -> None:
    cd = _staged(tmp_path)
    before = cd.read_text()
    remove_entries(cd, ["neverPresent"])  # no-op, no raise
    assert cd.read_text() == before


def test_remove_entries_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        remove_entries(tmp_path / "nope", ["endTime"])
