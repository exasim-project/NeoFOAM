# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Two solid blocks coupled Dirichlet-Neumann through non-matching faces (``multiRun``).

``cases/coupledBlocks``: ``left`` on [0, 1] (``k = 1``, 4 x 3 cells, 400 K at
x = 0, Neumann side) and ``right`` on [1, 2] (``k = 3``, 6 x 5 cells, 300 K at
x = 2, Dirichlet side), 3 against 5 interface faces mapped nearest-neighbor,
serial with Aitken around each ``solve_T``. A time step of 1e6 s is long
against the diffusion time, so each step is the steady profile: linear in each
block, the interface at ``k_l (400 - T_i) = k_r (T_i - 300)``, so 325 K. FV is
exact on a linear profile, so the tolerance 1e-6 K covers the linear solver
(1e-12) and the coupling's convergence (relative 1e-10) with margin.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neofoam.framework.coupling.kernels import nearest_neighbor
from solver.heatConductionNeoN._run_case import mesh_copy, result

T_INTERFACE = 325.0
# per cell in blockMesh order (x fastest): 400 - 75 x on left, 325 - 25 (x - 1) on right
STEADY_T = {
    "left": [390.625, 371.875, 353.125, 334.375] * 3,
    "right": [
        322.916666666667,
        318.75,
        314.583333333333,
        310.416666666667,
        306.25,
        302.083333333333,
    ]
    * 5,
}
ATOL_K = 1e-6


@pytest.fixture(scope="module")
def blocks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return result(mesh_copy("coupledBlocks", tmp_path_factory.mktemp("multiRun") / "case"), "multi")


@pytest.mark.parametrize("region", ["left", "right"])
def test_coupled_blocks_reach_the_steady_profile(blocks: dict[str, Any], region: str) -> None:
    data = blocks["regions"][region]

    np.testing.assert_allclose(
        data["T"], STEADY_T[region], rtol=0, atol=ATOL_K, err_msg=f"coupledBlocks/{region} T"
    )
    np.testing.assert_allclose(
        data["patches"]["interface"]["wall_T"],
        T_INTERFACE,
        rtol=0,
        atol=ATOL_K,
        err_msg=f"coupledBlocks/{region} interface wall T",
    )


def test_coupled_blocks_sides_are_the_dirichlet_neumann_pair(blocks: dict[str, Any]) -> None:
    patches = {region: data["patches"] for region, data in blocks["regions"].items()}

    assert [list(p) for p in patches.values()] == [["interface"], ["interface"]]
    assert patches["left"]["interface"]["type"] == "CoupledHeatFlux"
    assert patches["right"]["interface"]["type"] == "CoupledTemperature"


def test_coupled_blocks_interface_faces_do_not_match(blocks: dict[str, Any]) -> None:
    left = np.asarray(blocks["regions"]["left"]["patches"]["interface"]["points"])
    right = np.asarray(blocks["regions"]["right"]["patches"]["interface"]["points"])

    # face centres on x = 1, exact in binary up to round-off; only y = 0.5 is shared
    assert left.shape == (3, 3) and right.shape == (5, 3)
    np.testing.assert_allclose(left[:, 0], 1.0, rtol=0, atol=1e-12, err_msg="left interface x")
    np.testing.assert_allclose(right[:, 0], 1.0, rtol=0, atol=1e-12, err_msg="right interface x")
    assert np.sum(np.isclose(left[:, None, 1], right[None, :, 1])) == 1
    assert nearest_neighbor(left, right).tolist() == [0, 0, 1, 2, 2]
    assert nearest_neighbor(right, left).tolist() == [0, 2, 4]
    assert len(blocks["regions"]["left"]["patches"]["interface"]["wall_T"]) == 3
    assert len(blocks["regions"]["right"]["patches"]["interface"]["wall_T"]) == 5


def test_coupled_blocks_neumann_side_writes_the_wall_temperature_it_sends(
    blocks: dict[str, Any],
) -> None:
    left = blocks["regions"]["left"]["patches"]["interface"]

    # copied in memory: round-off only; on disk: writePrecision 12
    np.testing.assert_allclose(
        left["face_T"], left["wall_T"], rtol=0, atol=1e-9, err_msg="left interface face T"
    )
    np.testing.assert_allclose(
        left["written_T"], left["wall_T"], rtol=0, atol=1e-6, err_msg="left interface written T"
    )


@pytest.mark.parametrize("region", ["left", "right"])
def test_coupled_blocks_write_the_interface_temperature(
    blocks: dict[str, Any], region: str
) -> None:
    written = blocks["regions"][region]["patches"]["interface"]["written_T"]

    np.testing.assert_allclose(
        written, T_INTERFACE, rtol=0, atol=ATOL_K, err_msg=f"coupledBlocks/{region} written T"
    )


def test_coupled_blocks_aitken_converges_in_a_few_passes(blocks: dict[str, Any]) -> None:
    # right steps last and leads; the final pass included
    assert blocks["passes"] == {"right": [5, 6]}


def test_coupled_blocks_tutorial_is_this_case() -> None:
    tutorial = Path(__file__).parents[3] / "tutorials" / "heatConductionNeoN" / "coupledBlocks"
    case = Path(__file__).parent / "cases" / "coupledBlocks"
    files = sorted(path.relative_to(case) for path in case.rglob("*") if path.is_file())

    assert files and all(
        (tutorial / name).read_bytes() == (case / name).read_bytes() for name in files
    )
