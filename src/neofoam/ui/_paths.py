# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turn a wizard path field into an absolute directory path."""

from __future__ import annotations

from pathlib import Path


def _resolve_target(raw: str, label: str) -> Path:
    """Absolute path from the UI field ``raw``; raise ``ValueError`` if unusable.

    A blank or relative field would otherwise resolve to the directory the server
    was launched from and the wizard would write a case into it.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError(f"No {label} — type an absolute path first.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"The {label} must be an absolute path, got '{text}'.")
    return path.resolve()
