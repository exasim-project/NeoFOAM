# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``system/regionCoupling.yaml`` read into :class:`RegionCouplingConfig`.

``cases/coupled_slabs`` sets every entry, ``cases/region_coupling/minimal`` only
the required ones; every other ``cases/region_coupling/<name>`` breaks one rule.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.framework.coupling import NearestNeighborMapping
from neofoam.solver.multiRun.configs import RegionCouplingConfig

CASES = Path(__file__).parent / "cases"


def test_region_coupling_config_reads_every_entry() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES / "coupled_slabs")

    assert config.model_dump() == {
        "exchanges": [
            {
                "name": "interface",
                "sides": {"left": "right", "right": "left"},
                "mapping": {"type": "nearest-neighbor"},
            }
        ],
        "couplings": [
            {
                "name": "coupling",
                "scheme": "parallel",
                "maxIterations": 100,
                "exchanges": [
                    {"name": "interface", "acceleration": {"type": "aitken", "relaxation": 0.5}}
                ],
                "convergence": [{"data": "T", "type": "relative", "limit": 1e-12}],
                "wraps": {},
            }
        ],
        "transportTimeout": 600.0,
    }


def test_region_coupling_config_defaults_the_optional_entries() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES / "region_coupling" / "minimal")

    assert config.model_dump() == {
        "exchanges": [
            {
                "name": None,
                "sides": {"solid": "interface", "fluid": "interface"},
                "mapping": {"type": "nearest-neighbor"},
            }
        ],
        "couplings": [
            {
                "name": "thermal",
                "scheme": "serial",
                "maxIterations": 100,
                "exchanges": [{"name": "solid-fluid", "acceleration": None}],
                "convergence": [{"data": "T", "type": "relative", "limit": 1e-5}],
                "wraps": {},
            }
        ],
        "transportTimeout": 600.0,
    }


def test_region_coupling_config_reads_a_list_of_exchanges() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES.parent / "limitations" / "cases" / "chain")

    assert [(e.name, e.sides, type(e.mapping)) for e in config.exchanges] == [
        (None, {"a": "to_b", "b": "to_a"}, NearestNeighborMapping),
        (None, {"b": "to_c", "c": "to_b"}, NearestNeighborMapping),
    ]


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("unknown_scheme", r"couplings.0.scheme\n.*Input should be 'serial' or 'parallel'"),
        (
            "unknown_mapping",
            r"exchanges.0.mapping.mapping\n.*Input tag 'rbf' .* expected tags: 'identity', "
            r"'nearest-neighbor'",
        ),
        (
            "unknown_acceleration",
            r"acceleration.acceleration\n.*Input tag 'anderson' .* expected tags: 'constant', "
            r"'aitken'",
        ),
        (
            "acceleration_option_typo",
            r"acceleration.aitken.relaxation\n.*Field required.*\n.*\n.*acceleration.aitken."
            r"relax\n.*Extra inputs",
        ),
        (
            "relaxation_above_one",
            r"acceleration.constant.relaxation\n.*less than or equal to 1",
        ),
        ("unknown_convergence", r"convergence.0.convergence\n.*Input tag 'absolute' .* 'relative'"),
        (
            "convergence_option_typo",
            r"convergence.relative.limit\n.*Field required.*\n.*\n.*convergence.relative."
            r"lmit\n.*Extra inputs",
        ),
        ("no_convergence", r"couplings.0.convergence\n.*at least 1 item"),
        ("three_regions", r"exchanges.0.sides\n.*at most 2 items"),
        ("top_level_convergence", r"couplings\n.*Field required.*\n.*\nconvergence\n.*Extra"),
    ],
)
def test_region_coupling_config_rejects_an_invalid_file(case: str, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        RegionCouplingConfig.load(case_dir=CASES / "region_coupling" / case)
