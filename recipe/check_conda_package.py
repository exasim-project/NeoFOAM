#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# SPDX-License-Identifier: MIT

"""Smoke test for an installed NeoFOAM conda package.

Ported from src/NeoN/recipe/check_conda_package.py. Two deliberate differences:

* No ``__version__`` assertion. ``src/neofoam/__init__.py`` hardcodes ``0.0.1``
  while ``pyproject.toml`` says ``0.3.0``, so comparing either against
  ``PKG_VERSION`` would fail for reasons unrelated to packaging. Reported rather
  than worked around.
* The real assertion is that the native stack loaded. ``neofoam/__init__.py``
  catches ``ModuleNotFoundError`` and leaves ``neofoam_bindings`` as ``None``, so
  a plain ``import neofoam`` passes even when the extension is absent entirely.
"""

from __future__ import annotations

import pybFoam  # bundled in this package, not a separate dependency

import neofoam

# Imported at module scope rather than inside main() because ruff PLC0415 forbids the latter.
# It still carries the assertion: a load failure the guarded import in neofoam/__init__ would
# have hidden — an unresolved libnanobind.so from the bundled pybFoam, say — raises here.
import neofoam.neofoam_bindings as bindings


def main() -> None:
    if neofoam.neofoam_bindings is None:
        raise SystemExit(
            "neofoam imported but neofoam_bindings is None: the native extension is "
            "missing or failed to load (__init__ swallows ModuleNotFoundError)"
        )

    for name in ("create_adapter_run_time", "MeshAdapter", "RunTime"):
        if not hasattr(bindings, name):
            raise SystemExit(f"neofoam_bindings is missing {name}")

    # MeshAdapter declares Foam::fvMesh as its base class, and that type is registered by
    # pybFoam. Reaching it proves both extensions share one nanobind type registry, which is
    # the whole reason pybFoam is bundled rather than depended upon.
    if not issubclass(bindings.MeshAdapter, pybFoam.fvMesh):
        raise SystemExit(
            "MeshAdapter is not a subclass of pybFoam.fvMesh: the bindings and the bundled "
            "pybFoam are not sharing a nanobind type registry"
        )

    print(f"NeoFOAM conda package OK: {bindings.__file__}")


if __name__ == "__main__":
    main()
