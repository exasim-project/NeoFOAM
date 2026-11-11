# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``system/regionCoupling.yaml`` read into :class:`RegionCouplingConfig`.

``cases/coupled_slabs`` sets every entry (``exchanges``, a list);
``cases/region_coupling/minimal`` only the required ones (one exchange and
convergence), so the rest take their defaults. Every other
``cases/region_coupling/<name>`` breaks one rule of the file. The limitations
cases list several exchanges. Mapping, acceleration and convergence are
strategies of the framework's plugin families, selected by ``type``.
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

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
        "scheme": "parallel",
        "acceleration": {"type": "aitken", "relaxation": 0.5},
        "convergence": [{"data": "T", "type": "relative", "limit": 1e-12}],
        "maxIterations": 100,
        "couplings": None,
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
        "scheme": "parallel",
        "acceleration": None,
        "convergence": [{"data": "T", "type": "relative", "limit": 1e-5}],
        "maxIterations": None,
        "couplings": None,
        "transportTimeout": 600.0,
    }


def test_region_coupling_config_reads_a_list_of_exchanges() -> None:
    config = RegionCouplingConfig.load(case_dir=CASES.parent / "limitations" / "cases" / "chain")

    assert [(e.name, e.sides, e.mapping.type) for e in config.exchanges] == [
        (None, {"a": "to_b", "b": "to_a"}, "nearest-neighbor"),
        (None, {"b": "to_c", "c": "to_b"}, "nearest-neighbor"),
    ]


@pytest.mark.parametrize(
    ("error", "update"),
    [
        (
            r"acceleration.acceleration\n.*Input tag 'anderson' found using 'type' does not match "
            r"any of the expected tags: 'constant', 'aitken'",
            {"acceleration": {"type": "anderson", "relaxation": 0.5}},
        ),
        (
            r"convergence.0.convergence\n.*Input tag 'absolute' .* expected tags: 'relative'",
            {"convergence": [{"type": "absolute", "data": "x", "limit": 1}]},
        ),
        (
            r"exchanges.0.mapping.mapping\n.*Input tag 'rbf' .* expected tags: 'identity', "
            r"'nearest-neighbor'",
            {"exchanges": [{"sides": {"a": "p", "b": "q"}, "mapping": "rbf"}]},
        ),
        (
            r"acceleration.acceleration.aitken.relaxation\n.*Field required.*\n.*\n"
            r"acceleration.acceleration.aitken.relax\n.*Extra inputs",
            {"acceleration": {"type": "aitken", "relax": 0.5}},
        ),
        (
            r"convergence.0.convergence.relative.limit\n.*Field required.*\n.*\n"
            r"convergence.0.convergence.relative.lmit\n.*Extra inputs",
            {"convergence": [{"type": "relative", "data": "x", "lmit": 1}]},
        ),
    ],
    ids=[
        "unknown acceleration",
        "unknown convergence",
        "unknown mapping",
        "acceleration option",
        "convergence option",
    ],
)
def test_region_coupling_config_names_the_registered_types_and_options(
    error: str, update: dict[str, Any]
) -> None:
    data = {
        "exchanges": [{"sides": {"a": "p", "b": "q"}}],
        "convergence": [{"type": "relative", "data": "x", "limit": 1e-6}],
    }

    with pytest.raises(ValidationError, match=error):
        RegionCouplingConfig.model_validate({**data, **update})


@pytest.mark.parametrize(
    ("case", "entry"),
    [
        ("unknown_scheme", "scheme"),
        ("unknown_mapping", "mapping"),
        ("no_convergence", "convergence"),
        ("three_regions", "exchanges.0.sides"),
        (
            "relaxation_above_one",
            r"acceleration.acceleration.constant.relaxation\n.*less than or equal to 1",
        ),
    ],
)
def test_region_coupling_config_rejects_an_invalid_file(case: str, entry: str) -> None:
    with pytest.raises(ValidationError, match=entry):
        RegionCouplingConfig.load(case_dir=CASES / "region_coupling" / case)
