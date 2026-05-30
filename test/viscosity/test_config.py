# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for reading ``constant/transportProperties`` via BaseConfig.

Parametrized over every discovered case: the real OpenFOAM dictionary is read
and compared against the case's ``expected.yaml`` manifest, so the manifests can
never silently disagree with the shipped dicts. No dict content or expected
value is encoded in this module.
"""

import pytest

from neofoam.viscosity.config import TransportPropertiesConfig
from neofoam.viscosity.selection import model_name

from viscosity.conftest import CASES, Case


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_transport_properties_match_manifest(case: Case) -> None:
    cfg = TransportPropertiesConfig.load(case_dir=case.path)
    dumped = cfg.model_dump()
    # Compare only the keys the manifest declares — robust to optional-field
    # defaults; the fallback case's ``nu: null`` is still asserted explicitly.
    for key, expected in case.config.items():
        assert dumped[key] == expected
    assert model_name(cfg) == case.selection["model_name"]
