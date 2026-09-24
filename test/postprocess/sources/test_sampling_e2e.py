# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""End-to-end: what OpenFOAM cuts and interpolates for the surface sources.

**Integration, real OpenFOAM.** ``test_sampling.py`` pins the value layer with a
fake surface; this module pins the other half — that a declared plane really is
the plane the user asked for, that an iso-surface finds its level set, and that
``sample`` puts the field on those faces.

**Analytic expectations.** The two probe fields are seeded, not solved: ``p`` is
the cell-centre *x* coordinate and ``U = (x, y, 0)`` (see
``_sampling_driver.py``). That makes every expected number exact rather than a
tolerance around a solver result — the ``p == 0.05`` iso-surface *is* the plane
``x = 0.05``, whose area is the 0.1 x 0.01 m cross-section of the cavity, and a
``cell``-scheme sample of ``U`` on that plane is the value of each cut cell.

**Subprocess.** One ``Foam::Time`` per process, and the fixture already spends
this process's on ``blockMesh``, so the pipelines run in
``_sampling_driver.py`` and only their numbers come back as JSON.

**Tolerances.** ``_AREA_RTOL`` is the plan's 2 % budget for a cut area (OpenFOAM
in fact returns it to round-off). ``_SPEED_RTOL`` is looser than round-off
because the reference averages the *analytic* cell centres while blockMesh
computes its own by face decomposition; the difference measured on this mesh is
~4e-12, and 1e-8 keeps four decades of headroom.

**The mean is unweighted**, so it equals the average over the three cut cells
only because the cut splits each of them into the same number of faces (two, on
this uniform mesh). A mesh where that fails would need ``SurfIntegrate``
divided by the area instead.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from solver.incompressibleFluid.case_presets import lid_driven_cavity
from solver.incompressibleFluid.solved_case import CAVITY_TEMPLATE

from neofoam.tooling.casebuild import CaseDir, block_mesh, configs, from_template

_DRIVER = Path(__file__).parent / "_sampling_driver.py"

#: ``cases/cavity3x3``: a 0.1 x 0.1 x 0.01 m box of 3 x 3 x 1 cells, so a cut
#: normal to x has this area and the cell centres sit at these y.
_CROSS_SECTION = 0.1 * 0.01
_CELL_CENTRES_Y = (1 / 60, 3 / 60, 5 / 60)
_PLANE_X = 0.05

#: Mean of ``|U| = |(x, y, 0)|`` over the three cells the mid-plane cuts.
_MEAN_SPEED = sum(math.hypot(_PLANE_X, y) for y in _CELL_CENTRES_Y) / len(_CELL_CENTRES_Y)

#: See the module docstring.
_AREA_RTOL = 0.02
_SPEED_RTOL = 1e-8


@pytest.fixture(scope="module")
def sampled(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    """Every pipeline of ``_sampling_driver`` evaluated once on a meshed cavity."""
    case: CaseDir = (
        from_template(CAVITY_TEMPLATE) | configs(*lid_driven_cavity()) | block_mesh()
    ).build_at(tmp_path_factory.mktemp("sampling") / "cavity")
    out = case.path / "sampled.json"

    run = subprocess.run(
        [sys.executable, str(_DRIVER), str(case.path), str(out)],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "FOAM_SIGFPE": "false"},
    )
    assert run.returncode == 0, f"sampling driver aborted:\n{run.stdout[-2000:]}\n{run.stderr}"
    return dict(json.loads(out.read_text()))


def test_a_plane_cuts_the_cross_section_of_the_cavity(sampled: dict[str, float]) -> None:
    assert sampled["cut_area"] == pytest.approx(_CROSS_SECTION, rel=_AREA_RTOL)


def test_the_area_node_measures_the_same_cut(sampled: dict[str, float]) -> None:
    assert sampled["cut_area_via_area"] == pytest.approx(_CROSS_SECTION, rel=_AREA_RTOL)


def test_the_plane_cuts_the_three_cells_of_its_column_in_two(sampled: dict[str, float]) -> None:
    # Six faces of equal share — the premise of the unweighted mean below.
    assert sampled["mean_face_area"] == pytest.approx(_CROSS_SECTION / 6, rel=_AREA_RTOL)


def test_an_iso_surface_of_a_linear_field_is_the_plane_it_describes(
    sampled: dict[str, float],
) -> None:
    assert sampled["iso_area"] == pytest.approx(_CROSS_SECTION, rel=_AREA_RTOL)


def test_a_field_sampled_on_its_own_iso_surface_is_the_iso_value(
    sampled: dict[str, float],
) -> None:
    assert sampled["iso_p"] == pytest.approx(0.05, rel=_SPEED_RTOL)


@pytest.mark.parametrize(
    "table",
    ["mean_speed_from_source", "mean_speed_from_sample"],
    ids=["field_on_the_source", "sample_node"],
)
def test_the_mean_speed_on_the_plane_is_the_mean_over_the_cells_it_cuts(
    sampled: dict[str, float], table: str
) -> None:
    assert sampled[table] == pytest.approx(_MEAN_SPEED, rel=_SPEED_RTOL)
