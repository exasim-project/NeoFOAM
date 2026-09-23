# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the runnable-case scaffold (Allrun / Allclean)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from neofoam.ui import scaffold
from neofoam.ui.scaffold import ALLCLEAN_TEXT, scaffold_runnable_case

# Located from the test tree: CI installs a wheel, so the package is not in the checkout.
_REPO_ALLRUN = Path(__file__).resolve().parents[2] / "scripts" / "Allrun"


def test_scaffold_writes_executable_scripts(tmp_path):
    written = scaffold_runnable_case(tmp_path)
    allrun, allclean = tmp_path / "Allrun", tmp_path / "Allclean"

    assert set(written) == {allrun, allclean}
    for path in (allrun, allclean):
        assert path.is_file()
        assert os.access(path, os.X_OK), f"{path.name} is not executable"

    assert "neofoam solver incompressiblefluid" in allrun.read_text()
    assert allclean.read_text() == ALLCLEAN_TEXT
    assert "cleanCase0" in allclean.read_text()


@pytest.mark.parametrize(
    ("solver_name", "command"),
    [
        ("incompressibleFluid", "incompressiblefluid"),
        ("incompressibleFluidNeoN", "incompressiblefluidneon"),
        ("incompressibleVoF", "incompressiblevof"),
    ],
)
def test_scaffold_allrun_runs_the_named_solver(tmp_path, solver_name, command):
    scaffold_runnable_case(tmp_path, solver_name)

    solver_lines = [
        line for line in (tmp_path / "Allrun").read_text().splitlines() if "exec" in line
    ]
    assert solver_lines == [
        f'    exec "$NEOFOAM_PYTHON" -m neofoam.cli.app solver {command} "$@"',
        f'exec neofoam solver {command} "$@"',
    ]


def test_scaffold_default_allrun_is_the_template(tmp_path):
    scaffold_runnable_case(tmp_path)

    assert (tmp_path / "Allrun").read_text() == _REPO_ALLRUN.read_text()


def test_scaffold_is_idempotent(tmp_path):
    scaffold_runnable_case(tmp_path)
    first = (tmp_path / "Allrun").read_text()
    scaffold_runnable_case(tmp_path)  # second call must not error
    assert (tmp_path / "Allrun").read_text() == first


def test_embedded_allrun_matches_the_repo_template():
    assert scaffold._ALLRUN_FALLBACK == _REPO_ALLRUN.read_text()
