# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What a wizard path field resolves to: an absolute directory, or a refusal.

``HOME`` points at ``tmp_path`` so the ``~`` case has a literal expectation.
"""

from __future__ import annotations

import re

import pytest

from neofoam.ui._paths import _resolve_target


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        pytest.param("", "No target directory — type an absolute path first.", id="blank"),
        pytest.param("   ", "No target directory — type an absolute path first.", id="spaces"),
        pytest.param(
            "runs/case",
            "The target directory must be an absolute path, got 'runs/case'.",
            id="relative",
        ),
    ],
)
def test_resolve_target_refuses_a_field_that_is_not_an_absolute_path(raw, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _resolve_target(raw, "target directory")


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("~/case", id="home"),
        pytest.param(" {tmp}/case ", id="absolute"),
        pytest.param("{tmp}/runs/../case", id="unnormalized"),
    ],
)
def test_resolve_target_returns_the_absolute_directory(raw, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    path = _resolve_target(raw.format(tmp=tmp_path), "target directory")

    assert path == tmp_path / "case"
