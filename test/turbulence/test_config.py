# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for reading ``constant/turbulenceProperties`` via BaseConfig.

Parametrized over every discovered case: the real OpenFOAM dictionary (including
its nested ``RAS`` / ``LES`` sub-dictionaries) is read and compared against the
case's ``expected.yaml`` manifest, so the manifests can never silently disagree
with the shipped dicts, and the config is then re-saved and re-read so the
round-trip is pinned too. No dict content or expected value is encoded here.

The second half covers ``model_coefficients`` / ``load_with_coefficients`` — how a
closure's typed ``<Model>Coeffs`` is resolved out of that dictionary. Its inputs are
the very dictionaries the NeoN-vs-pybFoam parity run writes
(:func:`_parity_case.turbulence_config`): a bare ``<model>`` one that declares no
coefficients block, and a ``<model>Coeffs`` one that sets every coefficient off its
default plus one entry no closure declares. So the numbers pinned here are the numbers
both backends run. The *defaults* are literals: they are the values the closure classes
carry, and a test that recomputed them from those classes would prove nothing.
"""

from pathlib import Path
from typing import Any, Callable

import pytest

from neofoam.io import BaseConfig
from neofoam.tooling.casebuild import empty
from neofoam.turbulence.config import (
    TurbulencePropertiesConfig,
    load_with_coefficients,
    model_coefficients,
)
from neofoam.turbulence.models.kEpsilon import KEpsilonCoeffs
from neofoam.turbulence.models.kOmegaSST import KOmegaSSTCoeffs
from neofoam.turbulence.models.spalartAllmaras import SpalartAllmarasCoeffs
from neofoam.turbulence.selection import model_name
from turbulence._parity_case import COEFFS, turbulence_properties
from turbulence.conftest import CASES, Case

#: The typed coefficients config each closure declares.
COEFFS_TYPES: dict[str, type[BaseConfig]] = {
    "kEpsilon": KEpsilonCoeffs,
    "SpalartAllmaras": SpalartAllmarasCoeffs,
    "kOmegaSST": KOmegaSSTCoeffs,
}

#: The OpenFOAM defaults each closure class carries — what a dictionary with no
#: ``<model>Coeffs`` block must resolve to. Literals on purpose (see the module doc).
DEFAULTS: dict[str, dict[str, float]] = {
    "kEpsilon": {"Cmu": 0.09, "C1": 1.44, "C2": 1.92, "sigmak": 1.0, "sigmaEps": 1.3},
    "SpalartAllmaras": {
        "sigmaNut": 0.66666,
        "kappa": 0.41,
        "Cb1": 0.1355,
        "Cb2": 0.622,
        "Cw2": 0.3,
        "Cw3": 2.0,
        "Cv1": 7.1,
        "Cs": 0.3,
    },
    "kOmegaSST": {
        "alphaK1": 0.85,
        "alphaK2": 1.0,
        "alphaOmega1": 0.5,
        "alphaOmega2": 0.856,
        "gamma1": 5.0 / 9.0,
        "gamma2": 0.44,
        "beta1": 0.075,
        "beta2": 0.0828,
        "betaStar": 0.09,
        "a1": 0.31,
        "b1": 1.0,
        "c1": 10.0,
    },
}


@pytest.fixture
def model_case(tmp_path: Path) -> Callable[[str], Path]:
    """Materialize the case whose ``turbulenceProperties`` is the named dictionary."""

    def build(name: str) -> Path:
        return (empty() | turbulence_properties(name)).build_at(tmp_path / name).path

    return build


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_turbulence_properties_match_manifest_and_round_trip(case: Case, tmp_path: Path) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    dumped = cfg.model_dump()
    # Compare only the keys the manifest declares; nested sub-dictionaries (RAS /
    # LES) are compared wholesale, so the manifest fully pins each sub-config.
    for key, expected in case.config.items():
        assert dumped[key] == expected
    assert model_name(cfg) == case.selection["model_name"]

    cfg.save(case_dir=tmp_path)
    assert TurbulencePropertiesConfig.load(case_dir=tmp_path).model_dump() == dumped


@pytest.mark.parametrize("model", sorted(DEFAULTS))
def test_coefficients_default_to_openfoam_values(
    model: str, model_case: Callable[[str], Path]
) -> None:
    """A dictionary with no ``<model>Coeffs`` block leaves every default in place."""
    coeffs = load_with_coefficients(model_case(model), model, COEFFS_TYPES[model]).coeffs
    assert coeffs.model_dump() == DEFAULTS[model]


@pytest.mark.parametrize("model", sorted(COEFFS))
def test_case_coefficients_override_the_defaults(
    model: str, model_case: Callable[[str], Path]
) -> None:
    """Every entry of the case's ``<model>Coeffs`` block replaces the class default.

    ``notACoefficient`` is dropped, not rejected — OpenFOAM ignores it too, and
    tolerating it is load-bearing: ``RASProperties`` is ``extra="allow"``, so an
    unrecognised key reaches the resolution rather than failing validation, and a case
    that carries one must still run. The equality against the override table therefore
    also pins the resolved key *set*.

    The block resolves identically whether the ``RAS`` sub-dictionary arrives typed or
    raw: ``BaseConfig.load(validate=True)`` yields a ``RASProperties``, whereas the
    ``validate=False`` load ``ModelSpec.instantiate`` performs (and with it
    ``load_with_coefficients``) leaves it a mapping. Both reach ``model_coefficients``,
    so both must give the same coefficients.
    """
    case = model_case(f"{model}Coeffs")
    coeffs_type = COEFFS_TYPES[model]

    coeffs = load_with_coefficients(case, model, coeffs_type).coeffs
    assert coeffs.model_dump() == COEFFS[model]
    assert not hasattr(coeffs, "notACoefficient")

    validated: Any = TurbulencePropertiesConfig.load(case_dir=case)
    assert model_coefficients(validated, model, coeffs_type).model_dump() == COEFFS[model]
