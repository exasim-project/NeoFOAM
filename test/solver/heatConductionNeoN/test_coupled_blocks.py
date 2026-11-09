# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two solid blocks coupled Dirichlet-Neumann through non-matching faces (``multiRun``).

``cases/coupledBlocks``: block ``left`` on [0, 1] (``k = 1``, ``rho cp = 2``,
4 x 3 cells, 400 K at x = 0) and block ``right`` on [1, 2] (``k = 3``,
``rho cp = 0.5``, 6 x 5 cells, 300 K at x = 2), adiabatic walls. At x = 1 left is the Neumann side
(``coupledHeatFlux``), right the Dirichlet side (``coupledTemperature``); the
interface has 3 faces on the left and 5 on the right, mapped
``nearest-neighbor``; serial, Aitken, in one coupling node around each region's
``solve_T`` (a serial node leaves the time step outside). Each time step (1e6, long against the
diffusion time) is a steady solve: linear in each block, the interface at the
temperature the heat flux continuity gives,

    k_l (400 - T_i) / 1 = k_r (T_i - 300) / 1   ->   T_i = 325 K,

whatever ``rho cp``: the exchanged heat flux is ``k dT/dn``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from foamlib import FoamFile

from neofoam.framework.coupling.kernels import nearest_neighbor
from solver.heatConductionNeoN._run_case import cell_centres, mesh_copy, result

T_LEFT, T_RIGHT, K_LEFT, K_RIGHT = 400.0, 300.0, 1.0, 3.0
T_INTERFACE = (K_LEFT * T_LEFT + K_RIGHT * T_RIGHT) / (K_LEFT + K_RIGHT)


def _analytic(region: str, x: np.ndarray) -> np.ndarray:
    if region == "left":
        return T_LEFT + (T_INTERFACE - T_LEFT) * x
    return T_INTERFACE + (T_RIGHT - T_INTERFACE) * (x - 1.0)


@pytest.fixture(scope="module")
def blocks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    case_dir = mesh_copy("coupledBlocks", tmp_path_factory.mktemp("multiRun") / "case")
    computed = result(case_dir, "multi")
    for region, data in computed["regions"].items():
        data["x"] = cell_centres(case_dir / region)[:, 0]
    computed["case_dir"] = case_dir
    return computed


def test_interface_temperature_is_the_flux_continuity_one() -> None:
    assert T_INTERFACE == 325.0


@pytest.mark.parametrize("region", ["left", "right"])
def test_blocks_reach_the_piecewise_linear_profile(blocks: dict[str, Any], region: str) -> None:
    data = blocks["regions"][region]
    x = data["x"]
    # FV is exact on a linear profile: what is left is the linear solver's and the
    # coupling's tolerance
    np.testing.assert_allclose(data["T"], _analytic(region, x), rtol=0, atol=1e-6)
    interface = data["patches"]["interface"]
    np.testing.assert_allclose(interface["wall_T"], T_INTERFACE, rtol=0, atol=1e-6)


def test_the_sides_are_the_dirichlet_neumann_pair(blocks: dict[str, Any]) -> None:
    patches = {region: data["patches"] for region, data in blocks["regions"].items()}
    assert [list(p) for p in patches.values()] == [["interface"], ["interface"]]
    assert patches["left"]["interface"]["type"] == "CoupledHeatFlux"
    assert patches["right"]["interface"]["type"] == "CoupledTemperature"


def test_the_interface_faces_do_not_match_and_are_mapped(blocks: dict[str, Any]) -> None:
    left = np.asarray(blocks["regions"]["left"]["patches"]["interface"]["points"])
    right = np.asarray(blocks["regions"]["right"]["patches"]["interface"]["points"])
    # the face centres: on x = 1, 3 against 5; only the middle one (y = 0.5) is shared
    assert left.shape == (3, 3) and right.shape == (5, 3)
    np.testing.assert_allclose(left[:, 0], 1.0)
    np.testing.assert_allclose(right[:, 0], 1.0)
    assert np.sum(np.isclose(left[:, None, 1], right[None, :, 1])) == 1
    # each side's nearest partner faces: right takes left's 3 values on its 5 faces
    assert nearest_neighbor(left, right).tolist() == [0, 0, 1, 2, 2]
    assert nearest_neighbor(right, left).tolist() == [0, 2, 4]
    # what each side received has the size of its own patch
    assert len(blocks["regions"]["left"]["patches"]["interface"]["wall_T"]) == 3
    assert len(blocks["regions"]["right"]["patches"]["interface"]["wall_T"]) == 5


def test_the_written_interface_temperature_is_the_exchanged_one(blocks: dict[str, Any]) -> None:
    # the Neumann side (left) reconstructs its wall temperature from its heat flux;
    # it stores it as T's face values, so the written T holds what it sends
    left = blocks["regions"]["left"]["patches"]["interface"]
    np.testing.assert_allclose(left["face_T"], left["wall_T"], rtol=0, atol=1e-9)
    written = {
        region: np.asarray(
            FoamFile(blocks["case_dir"] / region / "2e+06" / "T")["boundaryField"]["interface"][
                "value"
            ],
            dtype=float,
        )
        for region in ("left", "right")
    }
    np.testing.assert_allclose(written["left"], left["face_T"], rtol=0, atol=1e-6)
    # converged, both sides write the interface temperature
    np.testing.assert_allclose(written["left"], T_INTERFACE, rtol=0, atol=1e-6)
    np.testing.assert_allclose(written["right"], T_INTERFACE, rtol=0, atol=1e-6)


def test_aitken_converges_each_time_step_in_a_few_passes(blocks: dict[str, Any]) -> None:
    # right steps last: it leads the group and owns the controller; the final pass included
    assert blocks["passes"] == {"right": [5, 6]}


def test_the_tutorial_is_this_case() -> None:
    tutorial = Path(__file__).parents[3] / "tutorials" / "heatConductionNeoN" / "coupledBlocks"
    case = Path(__file__).parent / "cases" / "coupledBlocks"
    files = sorted(path.relative_to(case) for path in case.rglob("*") if path.is_file())
    assert files and all(
        (tutorial / name).read_bytes() == (case / name).read_bytes() for name in files
    )
