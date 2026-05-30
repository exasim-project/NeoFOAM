# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for reading ``constant/turbulenceProperties`` via BaseConfig.

Parametrized over every discovered case: the real OpenFOAM dictionary (including
its nested ``RAS`` / ``LES`` sub-dictionaries) is read and compared against the
case's ``expected.yaml`` manifest, so the manifests can never silently disagree
with the shipped dicts. No dict content or expected value is encoded here.
"""

import pytest

from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.selection import model_name

from turbulence.conftest import CASES, Case


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_turbulence_properties_match_manifest(case: Case) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    dumped = cfg.model_dump()
    # Compare only the keys the manifest declares; nested sub-dictionaries (RAS /
    # LES) are compared wholesale, so the manifest fully pins each sub-config.
    for key, expected in case.config.items():
        assert dumped[key] == expected
    assert model_name(cfg) == case.selection["model_name"]
