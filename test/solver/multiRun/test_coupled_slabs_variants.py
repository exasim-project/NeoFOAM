# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""multiRun couples the slabs of ``cases/coupled_slabs`` as each variant of its coupling says.

``cases/coupled_slabs_variants/<name>`` is laid over the case. Every scheme and
acceleration reaches the steady profile of ``test_coupled_slabs`` within 1e-8 K
(limit 1e-12) but the slowest, parallel with constant relaxation 0.7: its error
contracts by ~0.81 per pass, so it stops at a relative change of 1e-10 (at ~400 K,
4e-8 K) within its 100 passes and leaves at most 0.81 / 0.19 times that, ~1.7e-7 K
(measured 7.3e-8 K); 2e-7 K bounds it.
"""

from pathlib import Path

import numpy as np
import pytest

from neofoam.solver.multiRun import run
from solver.multiRun.case_files import copy_case

EXPECTED = {
    "left": [390.625, 371.875, 353.125, 334.375],
    "right": [321.875, 315.625, 309.375, 303.125],
}


def _case(variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = copy_case("coupled_slabs", tmp_path / "case", f"coupled_slabs_variants/{variant}")
    monkeypatch.chdir(run_dir)


@pytest.mark.parametrize(
    ("variant", "atol_k"),
    [
        ("serial-aitken", 1e-8),
        ("serial-constant", 1e-8),
        ("parallel-aitken", 1e-8),
        ("parallel-constant", 2e-7),
    ],
)
def test_coupled_slabs_reaches_the_steady_profile(
    variant: str, atol_k: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case(variant, tmp_path, monkeypatch)

    regions = run()

    for region, expected in EXPECTED.items():
        np.testing.assert_allclose(
            regions[region].fields["T"],
            expected,
            rtol=0,
            atol=atol_k,
            err_msg=f"coupled_slabs {variant} region {region}",
        )


def test_coupled_slabs_convergence_on_data_no_side_sends_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case("convergence_on_U", tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match=r"convergence RelativeConvergence on U, which no port of the node sends"
    ):
        run()


def test_coupled_slabs_of_a_region_without_a_coupled_patch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case("no_coupled_side", tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match="exchange 'interface' connects right.left, which has no coupled instance"
    ):
        run()
