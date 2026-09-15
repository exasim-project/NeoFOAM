# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the packaged case-wizard notebook template + its writer.

``neofoam.agent.wizard_template`` ships the marimo wizard as a string and
scaffolds it to disk; the scaffolded file must be a valid, standalone marimo
notebook (it is what ``neofoam agent wizard`` writes).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from neofoam.agent import NOTEBOOK_TEMPLATE, write_wizard_notebook


def test_template_is_valid_python_marimo_notebook() -> None:
    # Parses as Python and is a marimo app (so ``marimo edit`` can open it).
    ast.parse(NOTEBOOK_TEMPLATE)
    assert "import marimo" in NOTEBOOK_TEMPLATE
    assert "app = marimo.App" in NOTEBOOK_TEMPLATE
    # Operates on the directory it is written to (scaffold-anywhere).
    assert "Path(__file__).resolve().parent" in NOTEBOOK_TEMPLATE


def test_write_wizard_notebook_writes_runnable_file(tmp_path: Path) -> None:
    path = write_wizard_notebook(tmp_path)
    assert path == tmp_path / "case_wizard.py"
    text = path.read_text()
    assert text == NOTEBOOK_TEMPLATE
    ast.parse(text)  # scaffolded file is valid Python


def test_write_wizard_notebook_custom_name_and_mkdir(tmp_path: Path) -> None:
    nested = tmp_path / "cases" / "cavity"
    path = write_wizard_notebook(nested, filename="wizard.py")
    assert path == nested / "wizard.py"
    assert path.exists()


def test_write_wizard_notebook_refuses_overwrite(tmp_path: Path) -> None:
    write_wizard_notebook(tmp_path)
    with pytest.raises(FileExistsError):
        write_wizard_notebook(tmp_path)
    # force overwrites without raising.
    path = write_wizard_notebook(tmp_path, force=True)
    assert path.exists()
