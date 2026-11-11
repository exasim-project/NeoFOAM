# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun couples the slabs of ``cases/coupled_slabs`` as each variant of its coupling says.

``cases/coupled_slabs_variants/<name>`` is laid over the case. Every scheme and
acceleration reaches the steady profile of ``test_multiRun_coupling``; the
slowest (parallel, constant 0.7) contracts by ~0.73 per pass, so a relative
change of 1e-12 (3e-10 K) leaves at most ~1e-9 K; 1e-8 K bounds every variant.
"""

import importlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from neofoam.framework.solver.registry import resolve_solver
from neofoam.solver.multiRun import run
from solver.multiRun.slab_solver import slab

CASES = Path(__file__).parent / "cases"
MULTI_RUN_MODULE = importlib.import_module("neofoam.solver.multiRun.multiRun")
EXPECTED = {
    "left": [390.625, 371.875, 353.125, 334.375],
    "right": [321.875, 315.625, 309.375, 303.125],
}


def _case(variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(CASES / "coupled_slabs", tmp_path / "case")
    shutil.copytree(
        CASES / "coupled_slabs_variants" / variant, tmp_path / "case", dirs_exist_ok=True
    )
    monkeypatch.setattr(
        MULTI_RUN_MODULE,
        "resolve_solver",
        lambda name: slab if name == "slab" else resolve_solver(name),
    )
    monkeypatch.chdir(tmp_path / "case")


@pytest.mark.parametrize(
    "variant", ["serial-aitken", "serial-constant", "parallel-aitken", "parallel-constant"]
)
def test_multiRun_region_coupling_reaches_the_steady_profile(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case(variant, tmp_path, monkeypatch)

    regions = run()

    for region, expected in EXPECTED.items():
        np.testing.assert_allclose(
            regions[region].fields["T"],
            expected,
            rtol=0,
            atol=1e-8,
            err_msg=f"coupled_slabs {variant} region {region}",
        )


def test_multiRun_region_coupling_convergence_on_data_no_side_sends_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case("convergence_on_U", tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match=r"convergence RelativeConvergence on U, which no side of the node sends"
    ):
        run()


def test_multiRun_region_coupling_of_a_region_without_a_coupled_patch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case("no_coupled_side", tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match="exchange 'interface' connects right.left, which has no coupled side"
    ):
        run()
