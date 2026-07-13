# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Subprocess field reader: dump a ``vol*Field`` internal field to ``.npy``.

Constructing a ``Foam::Time`` twice in one interpreter corrupts OpenFOAM per-process
global state — later reads come back as ``nan`` — so every read runs in its own
process. This module is the entry point :meth:`neofoam.tooling.casebuild.CaseDir.read_field`
spawns; run it as ``python -m neofoam.tooling.casebuild._reader <case> <time> <field> <out.npy>``.

The requested time directory is staged as ``0/`` in a temporary case so the field is
read where a freshly-constructed ``Time`` (which starts at ``startTime``) can see it —
the same trick the standalone comparison readers use.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pybFoam as pyf
from pybFoam import volScalarField, volVectorField


def _resolve_time_dir(case: Path, time: str) -> Path:
    """Return the time directory to read: the latest numeric dir, or a named one."""
    if time != "latest":
        return case / time
    numeric = [
        d for d in case.iterdir() if d.is_dir() and d.name.replace(".", "", 1).isdigit()
    ]
    if not numeric:
        raise FileNotFoundError(f"No numeric time directory in {case}")
    return max(numeric, key=lambda d: float(d.name))


def read_field(case: Path, time: str, name: str, out: Path) -> None:
    """Read *name*'s internal field at *time* and save it to *out* as ``.npy``.

    The ``np.save`` runs while the ``Time``/``fvMesh``/field are still alive:
    ``internalField()`` is a view into the field's buffer, and serializing it here
    (rather than returning it up the stack) copies the values before that buffer is
    freed with the mesh.
    """
    time_dir = _resolve_time_dir(case, time)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "case"
        shutil.copytree(case / "system", staged / "system")
        shutil.copytree(case / "constant", staged / "constant")
        shutil.copytree(time_dir, staged / "0")

        runtime = pyf.Time(str(staged.parent), staged.name)
        mesh = pyf.fvMesh(runtime)
        header = (staged / "0" / name).read_text()
        field: Union[volScalarField, volVectorField]
        if "volScalarField" in header:
            field = volScalarField.read_field(mesh, name)
        elif "volVectorField" in header:
            field = volVectorField.read_field(mesh, name)
        else:
            raise ValueError(f"Unsupported field type for {name}")
        np.save(out, np.asarray(field.internalField()))


def main() -> None:
    case, time, name, out = sys.argv[1:5]
    read_field(Path(case), time, name, Path(out))


if __name__ == "__main__":
    main()
