# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``$FOAM_CASE``: what makes OpenFOAM's ``<system>`` path tags resolve to the case."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pybFoam")

from neofoam.io.dictread import foam_case, set_foam_case  # noqa: E402

#: A case whose snappyHexMeshDict includes OpenFOAM's shipped .cfg, which ends in
#: ``#include "<system>/meshQualityDict"``.
_ETC_INCLUDE_CASE = Path(__file__).resolve().parents[1] / "ui" / "cases" / "etc_include"

#: Reading it aborts the process outright when the tag cannot resolve, so the read is
#: done in a child and judged by its exit code — an exception never arrives.
_READ = "import sys, pybFoam; pybFoam.dictionary.read(sys.argv[1])"


def _reads_in_a_child(*, with_case: bool) -> bool:
    env = dict(os.environ)
    env.pop("FOAM_CASE", None)
    if with_case:
        env["FOAM_CASE"] = str(_ETC_INCLUDE_CASE)
    done = subprocess.run(
        [sys.executable, "-c", _READ, str(_ETC_INCLUDE_CASE / "system" / "snappyHexMeshDict")],
        capture_output=True,
        env=env,
        check=False,
    )
    return done.returncode == 0


def test_a_case_dict_using_a_path_tag_needs_foam_case() -> None:
    # Without it the include resolves against OpenFOAM's own etc/ and the read dies.
    assert _reads_in_a_child(with_case=False) is False
    assert _reads_in_a_child(with_case=True) is True


def test_foam_case_sets_and_restores_the_variable() -> None:
    before = os.environ.get("FOAM_CASE")
    with foam_case(_ETC_INCLUDE_CASE):
        assert os.environ["FOAM_CASE"] == str(_ETC_INCLUDE_CASE)
    assert os.environ.get("FOAM_CASE") == before


def test_foam_case_restores_an_outer_value(tmp_path: Path) -> None:
    # The wizard sets a lasting one on load; a read of another case nests inside it.
    set_foam_case(tmp_path)
    try:
        with foam_case(_ETC_INCLUDE_CASE):
            assert os.environ["FOAM_CASE"] == str(_ETC_INCLUDE_CASE)
        assert os.environ["FOAM_CASE"] == str(tmp_path.resolve())
    finally:
        set_foam_case(None)
    assert "FOAM_CASE" not in os.environ


def test_set_foam_case_resolves_and_clears(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_foam_case(".")
    try:
        assert os.environ["FOAM_CASE"] == str(tmp_path.resolve())
    finally:
        set_foam_case(None)
    assert "FOAM_CASE" not in os.environ
