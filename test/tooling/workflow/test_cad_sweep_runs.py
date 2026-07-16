# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Task-12 acceptance: an exported CAD sweep actually plans under snakemake.

A CAD sweep composes ``cad × mesh`` into distinct ``meshes/{cad}__{mesh}`` mini
cases, so a ``snakemake -n`` dry run must plan a ``cad_geometry`` job per CAD
variant (previously the ``{mesh}``-keyed rules could not route ``cad_axis.of`` and
the plan raised). The dry run does NOT execute the shell, so no FreeCAD/foamcad is
needed — only that snakemake can build and resolve the DAG.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("snakemake")

from neofoam.tooling.workflow.sweep import export_sweep  # noqa: E402

_FCSTD = "/nonexistent/tube_bank.FCStd"  # never opened: -n does not run the shell


def test_cad_sweep_dry_run_plans_cad_geometry_per_variant(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    export_sweep(
        out,
        solver_name="incompressibleFluid",
        base_case=tmp_path / "base",
        dimensions={},
        classes={},
        cad={
            "cad": {
                "model": _FCSTD,
                "variants": {"R8": {"R_mm": 8.0}, "R10": {"R_mm": 10.0}},
            }
        },
    )

    proc = subprocess.run(
        [sys.executable, "-m", "snakemake", "-n", "-p"],
        cwd=str(out),
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    # The plan resolves (no KeyError from cad_axis.of under a {mesh}-keyed rule).
    assert proc.returncode == 0, combined[-3000:]
    # cad_geometry is planned once per composite cad × mesh mini-case.
    assert "cad_geometry" in combined
    assert "meshes/R8__base" in combined
    assert "meshes/R10__base" in combined
