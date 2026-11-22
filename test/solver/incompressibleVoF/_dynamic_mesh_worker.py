# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Run ``incompressibleVoF`` on a moving-mesh case and dump the mesh state it ended on.

One ``Foam::Time`` per process, so the run happens here and the test reads the
JSON. What it dumps is what the mesh-motion step is supposed to have produced:
whether the mesh moved at all, where its cell centres ended up, and the buoyancy
head ``gh`` that must have been rebuilt on those centres.

Usage: ``python _dynamic_mesh_worker.py <case-dir>``; writes
``<case-dir>/dynamic_mesh.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from neofoam.solver.incompressibleVoF import run

if __name__ == "__main__":
    case_dir = Path(sys.argv[1]).resolve()
    os.chdir(case_dir)
    ctx = run(["incompressibleVoF"])
    mesh = ctx.mesh
    (case_dir / "dynamic_mesh.json").write_text(
        json.dumps(
            {
                "dynamic": mesh.dynamic(),
                "moving": mesh.moving(),
                "end_time": float(ctx.models["runtime"].value()),
                "cell_centres": np.asarray(mesh.C().internalField()).tolist(),
                "gh": np.asarray(ctx.fields["gh"].internalField()).tolist(),
            },
            indent=1,
        )
    )
