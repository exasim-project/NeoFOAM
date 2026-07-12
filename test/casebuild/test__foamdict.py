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

from neofoam.casebuild._foamdict import apply_overrides

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
