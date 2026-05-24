# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""Helpers used by the documentation tutorials.

The gallery scripts under ``examples/tutorials/`` call ``clone_case``
so each tutorial starts from a clean copy of a bundled OpenFOAM case
without polluting the source tree.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

__all__ = ["clone_case", "cases_dir"]


def cases_dir() -> Path:
    """Return the bundled-tutorial-cases directory.

    Resolution order:
      1. ``NEOFOAM_CASES_DIR`` env var if set.
      2. ``<repo>/tutorials`` if importing from a source checkout.
    """
    env = os.environ.get("NEOFOAM_CASES_DIR")
    if env:
        return Path(env)
    repo_tutorials = Path(__file__).resolve().parent.parent.parent / "tutorials"
    if repo_tutorials.is_dir():
        return repo_tutorials
    raise FileNotFoundError(
        "Could not locate the tutorials directory. Set NEOFOAM_CASES_DIR "
        "to the absolute path of the repo's tutorials/ folder."
    )


def clone_case(
    name: str,
    dest: Optional[Path] = None,
    *,
    prefix: str = "neofoam_",
) -> Path:
    """Clone a bundled tutorial case and return the path to the copy.

    Parameters
    ----------
    name : str
        Sub-directory name under :func:`cases_dir`, e.g.
        ``"passive_scalar_pitzDaily"``.
    dest : Path, optional
        Where to put the working copy. If omitted, a fresh temporary
        directory is created (matching pybFoam's
        ``clone_case("cavity")`` API). When given, the case is placed
        at ``dest / name`` and any existing path there is removed.
    prefix : str
        Prefix for the auto-created tempdir when ``dest`` is omitted.

    Notes
    -----
    If the source case has a ``0.orig/`` template directory it is
    restored to ``0/`` in the clone — mirrors what an OpenFOAM
    ``Allrun`` would do before launching a solver.
    """
    src = cases_dir() / name
    if not src.is_dir():
        raise FileNotFoundError(f"No tutorial case named {name!r} at {src}")

    if dest is None:
        workdir = Path(tempfile.mkdtemp(prefix=prefix))
        target = workdir / name
    else:
        target = Path(dest) / name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src, target, symlinks=True)

    orig = target / "0.orig"
    if orig.is_dir():
        zero = target / "0"
        if zero.exists():
            shutil.rmtree(zero)
        shutil.copytree(orig, zero, symlinks=True)

    return target
