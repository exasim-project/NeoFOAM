# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The "Load case" directory browser: what it lists and where it opens."""

from __future__ import annotations

from pathlib import Path

from neofoam.ui.dir_picker import list_dirs, start_dir


def test_list_dirs_names_the_sub_directories_in_order(tmp_path: Path) -> None:
    for name in ("pitzDaily", "cavity", "damBreak"):
        (tmp_path / name).mkdir()
    (tmp_path / "controlDict").write_text("")  # a file is not somewhere to step into

    assert list_dirs(tmp_path) == [
        {"name": "cavity", "path": str(tmp_path / "cavity")},
        {"name": "damBreak", "path": str(tmp_path / "damBreak")},
        {"name": "pitzDaily", "path": str(tmp_path / "pitzDaily")},
    ]


def test_list_dirs_leaves_out_dot_directories(tmp_path: Path) -> None:
    # .git / .decomp-backups are never a case and would bury the ones that are.
    (tmp_path / ".git").mkdir()
    (tmp_path / "cavity").mkdir()

    assert [row["name"] for row in list_dirs(tmp_path)] == ["cavity"]


def test_list_dirs_of_an_unreadable_directory_is_empty(tmp_path: Path) -> None:
    # Stepping into one must not take the dialog (or the wizard) down.
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o000)
    try:
        assert list_dirs(closed) == []
    finally:
        closed.chmod(0o755)


def test_list_dirs_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert list_dirs(tmp_path / "gone") == []


def test_start_dir_reopens_on_the_loaded_case(tmp_path: Path) -> None:
    case = tmp_path / "cavity"
    case.mkdir()

    assert start_dir(str(case)) == case.resolve()


def test_start_dir_falls_back_to_the_cwd(tmp_path: Path, monkeypatch) -> None:
    # No target yet, or one that has since been removed: browse from where we are.
    monkeypatch.chdir(tmp_path)

    assert start_dir("") == Path.cwd()
    assert start_dir("   ") == Path.cwd()
    assert start_dir(str(tmp_path / "gone")) == Path.cwd()
