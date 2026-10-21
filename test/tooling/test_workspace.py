# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``Workspace`` path sandbox: relative-only resolution under a fixed root.

Covers the two contracts callers depend on — a path inside the root resolves to an
absolute path, and any escape (absolute input, ``..`` traversal, symlink out) raises
``CaseAccessError`` rather than reaching outside the root. One subprocess case guards a
separate promise of the package init: ``import neofoam.tooling`` stays stdlib-only and
pulls in no optional frontend extras (mcp/ui).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neofoam.tooling import CaseAccessError, Workspace


def test_import_pulls_no_optional_frontend_extras() -> None:
    """Importing the workspace sandbox drags in no trame (ui) / fastmcp (mcp) extra.

    pybFoam is a hard dependency the eager ``neofoam`` package import always pulls, so
    only the *optional* frontend extras are asserted absent. Checked in a clean
    subprocess: the shared pytest session imports the mcp/ui frontends (which do pull
    those), so only a fresh interpreter can prove ``import neofoam.tooling`` avoids them.
    """
    probe = (
        "import sys, neofoam.tooling\n"
        "heavy = [m for m in ('trame', 'fastmcp') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_case_access_error_is_a_value_error() -> None:
    """The typed error subclasses ValueError so existing callers stay non-breaking."""
    assert issubclass(CaseAccessError, ValueError)


def test_resolve_confines_a_relative_case_id_under_root(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    resolved = ws.resolve("cases/run1")
    assert resolved == (tmp_path / "cases" / "run1").resolve()
    assert resolved.is_absolute()


def test_resolve_allows_interior_dotdot_that_stays_inside(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    resolved = ws.resolve("a/../b")
    assert resolved == (tmp_path / "b").resolve()


def test_resolve_rejects_an_absolute_case_id(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    with pytest.raises(CaseAccessError):
        ws.resolve("/etc/passwd")


def test_resolve_rejects_a_dotdot_escape(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path / "root")
    (tmp_path / "root").mkdir()
    with pytest.raises(CaseAccessError):
        ws.resolve("../secret")


def test_resolve_rejects_a_symlink_that_escapes_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified")
    (root / "link").symlink_to(outside)  # interior symlink → outside
    ws = Workspace.at(root)
    with pytest.raises(CaseAccessError):
        ws.resolve("link/secret.txt")


@pytest.mark.parametrize("case_id", ["", "."])
def test_resolve_rejects_empty_or_dot_case_id(tmp_path: Path, case_id: str) -> None:
    # An empty/"." id normalizes to the root itself — never a real case, and a
    # silent whole-root grant. Pinned so a future _within tweak can't start
    # accepting it (nor escaping it).
    ws = Workspace.at(tmp_path)
    with pytest.raises(CaseAccessError):
        ws.resolve(case_id)


def test_resolve_error_message_uses_the_kind_label(tmp_path: Path) -> None:
    # ``kind`` distinguishes which path was rejected (e.g. source_dir vs case_dir).
    ws = Workspace.at(tmp_path)
    with pytest.raises(CaseAccessError, match="source_dir"):
        ws.resolve("/abs", kind="source_dir")


def test_resolve_existing_requires_the_path_to_exist(tmp_path: Path) -> None:
    ws = Workspace.at(tmp_path)
    with pytest.raises(CaseAccessError):
        ws.resolve_existing("no_such_case")


def test_resolve_existing_rejects_a_file_when_a_directory_is_required(
    tmp_path: Path,
) -> None:
    (tmp_path / "afile").write_text("x")
    ws = Workspace.at(tmp_path)
    with pytest.raises(CaseAccessError):
        ws.resolve_existing("afile")


def test_resolve_existing_returns_an_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "case").mkdir()
    ws = Workspace.at(tmp_path)
    assert ws.resolve_existing("case") == (tmp_path / "case").resolve()


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses the R_OK readability check",
)
def test_resolve_existing_rejects_an_unreadable_directory(tmp_path: Path) -> None:
    d = tmp_path / "locked"
    d.mkdir()
    d.chmod(0o000)
    ws = Workspace.at(tmp_path)
    try:
        with pytest.raises(CaseAccessError):
            ws.resolve_existing("locked")
    finally:
        d.chmod(0o755)  # let pytest clean up tmp_path
