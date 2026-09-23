# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What the "Load case" directory browser lists, as plain data.

The wizard browses the machine the *server* runs on: a web page cannot hand back a
real directory path (``<input type="file">`` yields file contents, not a location),
so the dialog only renders the rows these functions return.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["list_dirs", "start_dir"]


def start_dir(target: str) -> Path:
    """Where the browser opens: the current target when it is one, else the cwd.

    Reopening on the loaded case puts its siblings one step away, which is where the
    next case usually is::

        start_dir("/cases/cavity")  # -> Path("/cases/cavity")
    """
    text = (target or "").strip()
    if text:
        here = Path(text).expanduser()
        if here.is_dir():
            return here.resolve()
    return Path.cwd()


def list_dirs(directory: Path) -> list[dict[str, str]]:
    """``directory``'s sub-directories as ``{"name", "path"}`` rows, sorted by name.

    Dot-directories are left out (``.git``, ``.decomp-backups`` — never a case). A
    directory the server may not read lists as no rows at all rather than raising, so
    stepping into one cannot take the dialog (or the wizard) down.
    """
    try:
        found = sorted(p for p in directory.iterdir() if p.is_dir())
    except OSError:
        return []
    return [{"name": p.name, "path": str(p)} for p in found if not p.name.startswith(".")]
