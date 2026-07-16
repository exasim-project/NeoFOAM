# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Structural check of the hx-CAD example: the CAD sweep axis is exported.

Drives the wizard → SweepPanel → ``export_sweep`` exactly like
``playground/hx-cad/generate.py``, but into a tmp dir and WITHOUT FreeCAD: adding
a CAD dimension and exporting only records the model path + numeric variants and
materialises the workflow files — the parametric model is driven later, at
``snakemake`` run time (rule ``cad_geometry``), which this test does not execute.
So it verifies the plugin's sweep integration end to end (UI → workflow) with no
optional CAD dependency.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("trame")
pytest.importorskip("trame_flow")

from trame.app import get_server  # noqa: E402

from neofoam.agent.case_fill import case_spec_to_configs, load_case_from_disk  # noqa: E402
from neofoam.mcp.registry import resolve_solver  # noqa: E402
from neofoam.ui import build_app  # noqa: E402
from neofoam.ui import case_spec as cs  # noqa: E402

_SEED = Path(__file__).resolve().parents[1] / "workflow" / "cases" / "tube_bank"
# A placeholder .FCStd path — never opened (cad_geometry runs only under snakemake).
_FCSTD = "/nonexistent/tube_bank.FCStd"


def _seed_base(base: Path) -> None:
    base.mkdir(parents=True)
    for sub in ("system", "constant", "0"):
        src = _SEED / sub
        if src.is_dir():
            shutil.copytree(src, base / sub)


def test_hx_cad_export_wires_the_cad_axis(tmp_path):
    base = tmp_path / "base"
    _seed_base(base)

    server = build_app(server=get_server("hx_cad_example_test"), plugins=[])
    state, ctrl = server.state, server.controller
    state.target_dir = str(base)

    # Seed the forms from the base case + save (so the sweep has a case to clone).
    solver = resolve_solver("incompressibleFluid")
    configs = case_spec_to_configs(load_case_from_disk(base, solver=solver))
    by_key = {e.key: e for e in ctrl.get_entries()}
    for key, data in cs.configs_to_form_state(ctrl.get_entries(), configs).items():
        state[by_key[key].state_key] = data
    ctrl.save_case()
    assert state.scaffolded, state.save_report

    # Add the CAD dimension (the plugin's Add-to-sweep seam) + a 2nd radius variant.
    ctrl.sweep_add_cad_dimension(_FCSTD, {"R_mm": 8.0})
    dim = "dim:cad"
    ctrl.sweep_rename_buffer(dim, "R8")
    ctrl.sweep_variant_rename(dim)
    ctrl.sweep_variant_add(dim)
    ctrl.sweep_rename_buffer(dim, "R10")
    ctrl.sweep_variant_rename(dim)
    ctrl.sweep_variant_edit(dim, {"R_mm": 10.0})

    out = tmp_path / "sweep"
    state.sweep_out_dir = str(out)
    ctrl.sweep_export()
    assert not state.sweep_error, state.sweep_error

    # Two CAD cases.
    cases = (out / "sweep.csv").read_text().strip().splitlines()[1:]
    assert len(cases) == 2

    # The CAD variant payloads are materialised.
    assert json.loads((out / "configs" / "cad" / "R8.json").read_text()) == {
        "R_mm": 8.0
    }
    assert json.loads((out / "configs" / "cad" / "R10.json").read_text()) == {
        "R_mm": 10.0
    }

    # The Snakefile threads the model + the cad axis + the cad_geometry rule.
    snakefile = (out / "Snakefile").read_text()
    assert f'CAD_MODEL = "{_FCSTD}"' in snakefile
    assert 'cad_axis = space.keyed("cad"' in snakefile
    assert "cad_geometry.smk" in snakefile

    # params.yaml carries the cad axis.
    params = (out / "params.yaml").read_text()
    assert "cad:" in params and "R_mm" in params
