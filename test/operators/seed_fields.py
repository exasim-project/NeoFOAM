# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Seed deterministic analytic T and U fields into a staged case's ``0/`` dir.

Run as a standalone subprocess (``python seed_fields.py <case_dir>``) — it
constructs a ``Foam::Time``, which must happen exactly once per process.

The fields are smooth, non-symmetric functions of the cell centres so every
operator produces a non-trivial result that is bit-identical input for both
backends (each backend reads the very same bytes from ``0/``). ``0/phi`` is
removed so both backends derive the flux from the same ``0/U`` via
``fvc::flux(U)``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pybFoam as pyf


def _replace_internal_field(path: Path, body: str) -> None:
    content = path.read_text()
    new, n_replaced = re.subn(
        r"internalField[^;]*;", f"internalField   {body};", content, count=1, flags=re.S
    )
    if n_replaced != 1:
        raise ValueError(f"no internalField entry replaced in {path}")
    path.write_text(new)


def _scalar_body(values: np.ndarray) -> str:
    entries = "\n".join(repr(float(v)) for v in values)
    return f"nonuniform List<scalar>\n{len(values)}\n(\n{entries}\n)"


def _vector_body(values: np.ndarray) -> str:
    entries = "\n".join(
        f"({float(v[0])!r} {float(v[1])!r} {float(v[2])!r})" for v in values
    )
    return f"nonuniform List<vector>\n{len(values)}\n(\n{entries}\n)"


def seed_case(case_dir: Path) -> None:
    import os

    os.chdir(case_dir)
    runtime = pyf.Time(pyf.argList(["seedFields"]))
    mesh = pyf.fvMesh(runtime)
    centres = np.asarray(mesh.C().internalField())

    span = centres.max(axis=0) - centres.min(axis=0)
    scale = float(span.max())
    x, y, z = (centres[:, i] / scale for i in range(3))

    # Non-symmetric and with nonzero in-plane divergence (U_x must vary with x
    # and U_y with y, otherwise div(phi) degenerates to round-off on 2D meshes).
    t_values = 2.0 + np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * z
    u_values = np.stack(
        [
            1.0 + np.sin(np.pi * y) + 0.5 * np.sin(np.pi * x),
            0.5 + np.cos(np.pi * x) + 0.5 * np.cos(np.pi * y),
            0.1 + 0.2 * np.sin(np.pi * z),
        ],
        axis=1,
    )

    zero = case_dir / "0"
    _replace_internal_field(zero / "T", _scalar_body(t_values))
    _replace_internal_field(zero / "U", _vector_body(u_values))
    (zero / "phi").unlink(missing_ok=True)


def main() -> None:
    seed_case(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
