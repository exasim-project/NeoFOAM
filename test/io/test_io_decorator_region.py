# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Configs of a multi-region case, laid out as in OpenFOAM.org.

A region directory follows the top directory (``constant/left/...``), so one
config class serves every region and the single-region case is the same call
without a region. The case ``cases/regions`` holds one value per region and a
different one at the single-region path, so a load from the wrong file fails.
"""

from pathlib import Path

import pytest

from neofoam.io import YAML, BaseConfig, IOStrategy

CASE = Path(__file__).parent / "cases" / "regions"


@IOStrategy(YAML("constant/conductivity.yaml"))
class ConductivityConfig(BaseConfig):
    conductivity: float


@pytest.mark.parametrize(("region", "expected"), [("left", 2.0), ("right", 3.0), (None, 1.0)])
def test_load_reads_the_region_directory(region: str | None, expected: float) -> None:
    loaded = ConductivityConfig.load(case_dir=CASE, region=region)

    assert loaded.conductivity == expected


def test_save_writes_the_region_directory(tmp_path: Path) -> None:
    ConductivityConfig(conductivity=4.0).save(case_dir=tmp_path, region="left")

    saved = tmp_path / "constant" / "left" / "conductivity.yaml"
    assert ConductivityConfig.load(case_dir=saved).conductivity == 4.0


@pytest.mark.parametrize(("region", "conductivity"), [("left", 4.0), ("right", 5.0), (None, 6.0)])
def test_save_and_load_round_trip_per_region(
    tmp_path: Path, region: str | None, conductivity: float
) -> None:
    ConductivityConfig(conductivity=conductivity).save(case_dir=tmp_path, region=region)

    assert ConductivityConfig.load(case_dir=tmp_path, region=region).conductivity == conductivity


def test_load_applies_the_region_to_a_file_override() -> None:
    loaded = ConductivityConfig.load(
        case_dir=CASE, file="constant/conductivity.yaml", region="right"
    )

    assert loaded.conductivity == 3.0


def test_region_needs_a_top_directory() -> None:
    with pytest.raises(ValueError, match="no top directory"):
        ConductivityConfig.load(case_dir=CASE, file="conductivity.yaml", region="left")
