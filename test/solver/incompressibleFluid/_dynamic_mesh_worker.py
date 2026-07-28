# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run ``incompressibleFluid`` on the moving duct and dump the state it ended on.

One ``Foam::Time`` per process, so the run happens here and the test reads the
JSON. What it dumps is what the mesh-motion chain is supposed to have produced:
which mesh class was selected, whether it moved, where its cells ended up, the
face flux it left behind (relative to the mesh motion) and the face velocity
``Uf`` that flux was corrected against.

Usage: ``python _dynamic_mesh_worker.py <case-dir>``; writes
``<case-dir>/dynamic_mesh.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from neofoam.solver.incompressibleFluid import run


def _internal(field: Optional[Any]) -> Optional[list[Any]]:
    return None if field is None else np.asarray(field.internalField()).tolist()


if __name__ == "__main__":
    case_dir = Path(sys.argv[1]).resolve()
    os.chdir(case_dir)
    ctx = run(["incompressibleFluid"])
    mesh = ctx.mesh
    (case_dir / "dynamic_mesh.json").write_text(
        json.dumps(
            {
                "mesh_type": type(mesh).__name__,
                "dynamic": mesh.dynamic(),
                "moving": mesh.moving(),
                "end_time": float(mesh.time().value()),
                "dynamic_mesh_controls": ctx.models["dynamic_mesh_controls"],
                "cell_centres": _internal(mesh.C()),
                "phi": _internal(ctx.fields["phi"]),
                "Uf": _internal(ctx.models["Uf"]),
            },
            indent=1,
        )
    )
