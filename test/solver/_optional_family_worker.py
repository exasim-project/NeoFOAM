# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Bind the example plugin family to one solver, load and run the case in the working directory.

Usage: ``python _optional_family_worker.py <solver>``; writes ``plugin_family.json``.
"""

import importlib
import json
import sys
from pathlib import Path

from _optional_family_plugin import radiationModel

from neofoam.framework.solver.configurations import model_catalog

if __name__ == "__main__":
    solver_name = sys.argv[1]
    solver_package = importlib.import_module(f"neofoam.solver.{solver_name}")
    create_fields = importlib.import_module(f"neofoam.solver.{solver_name}.create_fields")
    solver_spec = getattr(solver_package, solver_name)
    solver_spec.models(radiationModel)

    load_result = create_fields.create_init(Path.cwd()).run_load()
    loaded = [model.name for model in load_result.optional_models]
    catalog = [entry.name for entry in model_catalog(solver_spec)]
    solver_package.run([solver_name])

    log = Path("radiation_steps.log")
    calls = len(log.read_text().splitlines()) if log.exists() else 0
    Path("plugin_family.json").write_text(
        json.dumps({"loaded": loaded, "catalog": catalog, "calls": calls})
    )
