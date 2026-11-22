# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for reading ``constant/transportProperties`` via BaseConfig.

Parametrized over every discovered case: the real OpenFOAM dictionary is read
and compared against the case's ``expected.yaml`` manifest, so the manifests can
never silently disagree with the shipped dicts. No dict content or expected
value is encoded in this module.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.viscosity.config import TransportPropertiesConfig
from neofoam.viscosity.selection import model_name
from viscosity.conftest import CASES, Case

#: A Newtonian dictionary with the ``nu`` entry left out (kept outside ``cases/``
#: so the valid-case discovery never picks it up).
NEWTONIAN_WITHOUT_NU = Path(__file__).resolve().parent / "invalid_cases" / "newtonianWithoutNu"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_transport_properties_match_manifest(case: Case) -> None:
    cfg = TransportPropertiesConfig.load(case_dir=case.path)
    dumped = cfg.model_dump()
    # Compare only the keys the manifest declares — robust to optional-field
    # defaults; the fallback case's ``nu: null`` is still asserted explicitly.
    for key, expected in case.config.items():
        assert dumped[key] == expected
    assert model_name(cfg) == case.selection["model_name"]


def test_newtonian_without_nu_fails_validation() -> None:
    with pytest.raises(ValidationError, match="Newtonian.*requires 'nu'"):
        TransportPropertiesConfig.load(case_dir=NEWTONIAN_WITHOUT_NU)


@pytest.mark.parametrize(
    ("data", "missing"),
    [
        ({"transportModel": "Newtonian"}, ["'nu' is a required property"]),
        ({}, ["'nu' is a required property"]),  # transportModel defaults to Newtonian
        ({"transportModel": "Newtonian", "nu": 1e-05}, []),
        ({"transportModel": "CrossPowerLaw"}, []),
    ],
)
def test_json_schema_requires_nu_only_for_newtonian(
    data: dict[str, object], missing: list[str]
) -> None:
    """The published schema carries the validator's rule, so a form flags it before save."""
    jsonschema = pytest.importorskip("jsonschema")  # transitive via the `mcp` extra
    validator = jsonschema.Draft202012Validator(TransportPropertiesConfig.model_json_schema())
    assert [error.message for error in validator.iter_errors(data)] == missing
