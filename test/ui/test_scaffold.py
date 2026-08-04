# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the runnable-case scaffold (Allrun / Allclean)."""

from __future__ import annotations

import os

from neofoam.ui.scaffold import (
    ALLCLEAN_TEXT,
    allrun_template_path,
    scaffold_runnable_case,
)


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


def test_scaffold_is_idempotent(tmp_path):
    scaffold_runnable_case(tmp_path)
    first = (tmp_path / "Allrun").read_text()
    scaffold_runnable_case(tmp_path)  # second call must not error
    assert (tmp_path / "Allrun").read_text() == first


def test_allrun_template_resolves_in_source_checkout():
    template = allrun_template_path()
    assert template.name == "Allrun"
    assert template.is_file()  # repo scripts/Allrun exists
