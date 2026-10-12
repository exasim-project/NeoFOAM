# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``system/regionCoupling.yaml`` read into :class:`RegionCouplingConfig`.

``cases/coupled_slabs`` sets every entry; ``cases/region_coupling/minimal`` only
the required ones (interface, convergence), so the rest take their defaults.
Every other ``cases/region_coupling/<name>`` breaks one rule of the file.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.solver.multiRun.configs import RegionCouplingConfig

CASES = Path(__file__).parent / "cases"


def test_region_coupling_config_reads_every_entry() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES / "coupled_slabs")

    assert config.model_dump() == {
        "interface": {"left": "right", "right": "left"},
        "scheme": "serial",
        "mapping": "nearest-neighbor",
        "acceleration": {"type": "aitken", "relaxation": 0.5},
        "convergence": [{"type": "relative", "data": "T", "limit": 1e-12}],
        "maxIterations": 100,
    }


def test_region_coupling_config_defaults_the_optional_entries() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES / "region_coupling" / "minimal")

    assert config.model_dump() == {
        "interface": {"solid": "interface", "fluid": "interface"},
        "scheme": "serial",
        "mapping": "nearest-neighbor",
        "acceleration": {"type": "constant", "relaxation": 1.0},
        "convergence": [{"type": "relative", "data": "T", "limit": 1e-5}],
        "maxIterations": 100,
    }


@pytest.mark.parametrize(
    ("case", "entry"),
    [
        ("unknown_scheme", "scheme"),
        ("unknown_mapping", "mapping"),
        ("no_convergence", "convergence"),
        ("three_regions", "interface"),
        ("relaxation_above_one", "acceleration.relaxation"),
    ],
)
def test_region_coupling_config_rejects_an_invalid_file(case: str, entry: str) -> None:
    with pytest.raises(ValidationError, match=entry):
        RegionCouplingConfig.load(case_dir=CASES / "region_coupling" / case)
