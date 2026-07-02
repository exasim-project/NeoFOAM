# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Stage 1 (neofoam side): drive STL extraction across the conda-env boundary.

FreeCAD + foamcadagent live in a *different* conda env (py3.11) than neofoam
(py3.12), so extraction cannot be an in-process import. This module shells out
to :mod:`neofoam.e2e._extract_driver` in that env, then loads and validates the
``manifest.json`` it wrote -- the case directory is the cross-env contract.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from neofoam.e2e.manifest import PatchManifest

_DRIVER = Path(__file__).with_name("_extract_driver.py")


def _build_command(
    model_path: Path,
    case_dir: Path,
    scale: float,
    *,
    conda_env: str,
    conda_bin: str,
    python_bin: str | None,
) -> list[str]:
    """Build the subprocess argv that runs the driver in the FreeCAD env.

    ``python_bin`` (a direct interpreter path) bypasses conda; otherwise the
    driver is launched via ``conda run -n <conda_env>``.
    """
    driver_args = [str(_DRIVER), str(model_path), str(case_dir), "--scale", str(scale)]
    if python_bin:
        return [python_bin, *driver_args]
    return [conda_bin, "run", "--no-capture-output", "-n", conda_env, "python", *driver_args]


def extract_stl(
    model_path: str | Path,
    case_dir: str | Path,
    *,
    scale: float = 0.001,
    conda_env: str = "foamcadagent",
    conda_bin: str = "conda",
    python_bin: str | None = None,
) -> PatchManifest:
    """Extract named STL patches from ``model_path`` into ``case_dir``.

    Runs the FreeCAD-env driver, then returns the validated
    :class:`PatchManifest` it wrote to ``<case_dir>/manifest.json``.

    The FreeCAD env name is configurable via the ``conda_env`` argument or the
    ``NEOFOAM_CAD_ENV`` environment variable; ``conda_bin`` / ``NEOFOAM_CONDA``
    locate the conda executable. Pass ``python_bin`` to run a specific
    interpreter directly and skip conda entirely.
    """
    model_path = Path(model_path)
    case_dir = Path(case_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"CAD model not found: {model_path}")
    case_dir.mkdir(parents=True, exist_ok=True)

    conda_env = os.environ.get("NEOFOAM_CAD_ENV", conda_env)
    conda_bin = os.environ.get("NEOFOAM_CONDA", conda_bin)
    cmd = _build_command(
        model_path, case_dir, scale, conda_env=conda_env, conda_bin=conda_bin, python_bin=python_bin
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "STL extraction driver failed "
            f"(exit {result.returncode}).\ncmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    manifest_path = case_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"driver exited 0 but wrote no manifest at {manifest_path}.\nstderr:\n{result.stderr}"
        )
    return PatchManifest.load(manifest_path)
