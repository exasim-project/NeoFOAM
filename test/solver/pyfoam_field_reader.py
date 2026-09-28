# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Read OpenFOAM ``vol*Field`` internal fields via pybFoam and dump them as ``.npy``.

Run as a standalone subprocess (``python pyfoam_field_reader.py <case> <time_dir>
<out_dir> <field> ...``). Constructing a ``Foam::Time`` pulls in OpenFOAM
per-process global state that is corrupted by a second construction in the same
interpreter — subsequent reads come back as nan — so every read gets its own
process with exactly one ``Time``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pybFoam as pyf
from pybFoam import volScalarField, volVectorField


def read_fields(case: Path, time_dir: Path, out_dir: Path, field_names: list[str]) -> None:
    """Stage ``time_dir`` as ``0/`` and dump each field's internal field to ``.npy``."""
    staged = out_dir / "case"
    shutil.copytree(case / "system", staged / "system")
    shutil.copytree(case / "constant", staged / "constant")
    shutil.copytree(time_dir, staged / "0")
    os.chdir(staged)

    runtime = pyf.Time(pyf.argList(["test"]))
    mesh = pyf.fvMesh(runtime)
    for name in field_names:
        # The FoamFile header is ASCII even when `writeFormat binary`, but the
        # payload then is not — decode leniently and only sniff the header.
        header = (staged / "0" / name).read_bytes()[:2048].decode("utf-8", "replace")
        if "volScalarField" in header:
            field = volScalarField.read_field(mesh, name)
        elif "volVectorField" in header:
            field = volVectorField.read_field(mesh, name)
        else:
            raise ValueError(f"Unknown field type for {name}")
        np.save(out_dir / f"{name}.npy", np.asarray(field.internalField()))


def main() -> None:
    case, time_dir, out_dir = (Path(p) for p in sys.argv[1:4])
    read_fields(case, time_dir, out_dir, sys.argv[4:])


if __name__ == "__main__":
    main()
