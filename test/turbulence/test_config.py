# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for reading ``constant/turbulenceProperties`` via BaseConfig.

Parametrized over every discovered case: the real OpenFOAM dictionary (including
its nested ``RAS`` / ``LES`` sub-dictionaries) is read and compared against the
case's ``expected.yaml`` manifest, so the manifests can never silently disagree
with the shipped dicts. No dict content or expected value is encoded here.

The second half covers ``model_coefficients`` / ``load_with_coefficients`` — how a
closure's typed ``<Model>Coeffs`` is resolved out of that dictionary. Its inputs
are the ``parity_models/`` dictionaries the NeoN-vs-pybFoam parity run already
uses (a bare ``<model>`` one that declares no coefficients block, and a
``<model>Coeffs`` one that sets every coefficient off its default plus one entry
no closure declares), so the numbers pinned here are the numbers both backends
run. The expected values are literals: they are the defaults the closure classes
carry, and a test that recomputed them from those classes would prove nothing.
"""

from pathlib import Path
from typing import Any

import pytest

from neofoam.io import BaseConfig
from neofoam.turbulence.config import (
    TurbulencePropertiesConfig,
    load_with_coefficients,
    model_coefficients,
)
from neofoam.turbulence.models.kEpsilon import KEpsilonCoeffs
from neofoam.turbulence.models.kOmegaSST import KOmegaSSTCoeffs
from neofoam.turbulence.models.spalartAllmaras import SpalartAllmarasCoeffs
from neofoam.turbulence.selection import model_name
from turbulence.conftest import CASES, Case

_PARITY_MODELS = Path(__file__).parent / "parity_models"

#: ``(dictionary directory, model name, coeffs class, the coefficients it resolves to)``.
#: These dictionaries declare no ``<model>Coeffs`` block, so every field keeps the
#: OpenFOAM default the closure class carries.
DEFAULT_CASES = [
    (
        "kEpsilon",
        "kEpsilon",
        KEpsilonCoeffs,
        {"Cmu": 0.09, "C1": 1.44, "C2": 1.92, "sigmak": 1.0, "sigmaEps": 1.3},
    ),
    (
        "SpalartAllmaras",
        "SpalartAllmaras",
        SpalartAllmarasCoeffs,
        {
            "sigmaNut": 0.66666,
            "kappa": 0.41,
            "Cb1": 0.1355,
            "Cb2": 0.622,
            "Cw2": 0.3,
            "Cw3": 2.0,
            "Cv1": 7.1,
            "Cs": 0.3,
        },
    ),
    (
        "kOmegaSST",
        "kOmegaSST",
        KOmegaSSTCoeffs,
        {
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
    ),
]

#: Same shape, for the dictionaries whose ``<model>Coeffs`` block sets every
#: coefficient off its default (and adds ``notACoefficient``, which is ignored).
OVERRIDE_CASES = [
    (
        "kEpsilonCoeffs",
        "kEpsilon",
        KEpsilonCoeffs,
        {"Cmu": 0.085, "C1": 1.42, "C2": 1.68, "sigmak": 1.2, "sigmaEps": 1.11},
    ),
    (
        "SpalartAllmarasCoeffs",
        "SpalartAllmaras",
        SpalartAllmarasCoeffs,
        {
            "sigmaNut": 0.7,
            "kappa": 0.42,
            "Cb1": 0.14,
            "Cb2": 0.6,
            "Cw2": 0.32,
            "Cw3": 2.1,
            "Cv1": 7.0,
            "Cs": 0.35,
        },
    ),
    (
        "kOmegaSSTCoeffs",
        "kOmegaSST",
        KOmegaSSTCoeffs,
        {
            "alphaK1": 0.8,
            "alphaK2": 1.1,
            "alphaOmega1": 0.55,
            "alphaOmega2": 0.9,
            "gamma1": 0.52,
            "gamma2": 0.46,
            "beta1": 0.08,
            "beta2": 0.09,
            "betaStar": 0.085,
            "a1": 0.32,
            "b1": 1.1,
            "c1": 9.0,
        },
    ),
]


def _dict_path(dict_name: str) -> Path:
    """The shipped ``turbulenceProperties`` of a parity model, as an explicit file."""
    return _PARITY_MODELS / dict_name / "turbulenceProperties"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_turbulence_properties_match_manifest(case: Case) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    dumped = cfg.model_dump()
    # Compare only the keys the manifest declares; nested sub-dictionaries (RAS /
    # LES) are compared wholesale, so the manifest fully pins each sub-config.
    for key, expected in case.config.items():
        assert dumped[key] == expected
    assert model_name(cfg) == case.selection["model_name"]


@pytest.mark.parametrize(
    ("dict_name", "model", "coeffs_type", "expected"),
    DEFAULT_CASES,
    ids=[c[0] for c in DEFAULT_CASES],
)
def test_coefficients_default_to_openfoam_values(
    dict_name: str, model: str, coeffs_type: type[BaseConfig], expected: dict[str, float]
) -> None:
    """A dictionary with no ``<model>Coeffs`` block leaves every default in place."""
    coeffs = load_with_coefficients(_dict_path(dict_name), model, coeffs_type).coeffs
    assert coeffs.model_dump() == expected


@pytest.mark.parametrize(
    ("dict_name", "model", "coeffs_type", "expected"),
    OVERRIDE_CASES,
    ids=[c[0] for c in OVERRIDE_CASES],
)
def test_case_coefficients_override_the_defaults(
    dict_name: str, model: str, coeffs_type: type[BaseConfig], expected: dict[str, float]
) -> None:
    """Every entry of the case's ``<model>Coeffs`` block replaces the class default."""
    coeffs = load_with_coefficients(_dict_path(dict_name), model, coeffs_type).coeffs
    assert coeffs.model_dump() == expected


@pytest.mark.parametrize(
    ("dict_name", "model", "coeffs_type", "expected"),
    OVERRIDE_CASES,
    ids=[c[0] for c in OVERRIDE_CASES],
)
def test_undeclared_coefficient_entry_is_ignored(
    dict_name: str, model: str, coeffs_type: type[BaseConfig], expected: dict[str, float]
) -> None:
    """``notACoefficient`` is dropped, not rejected — OpenFOAM ignores it too.

    Tolerating it is load-bearing: ``RASProperties`` is ``extra="allow"``, so an
    unrecognised key reaches here rather than failing validation, and a case that
    carries one must still run.
    """
    coeffs = load_with_coefficients(_dict_path(dict_name), model, coeffs_type).coeffs
    assert not hasattr(coeffs, "notACoefficient")
    assert set(coeffs.model_dump()) == set(expected)


@pytest.mark.parametrize(
    ("dict_name", "model", "coeffs_type", "expected"),
    OVERRIDE_CASES,
    ids=[c[0] for c in OVERRIDE_CASES],
)
def test_coefficients_resolve_from_a_validated_config(
    dict_name: str, model: str, coeffs_type: type[BaseConfig], expected: dict[str, float]
) -> None:
    """The ``RAS`` block resolves identically as a ``RASProperties`` or a plain dict.

    ``BaseConfig.load(validate=True)`` yields the typed sub-config, whereas the
    ``validate=False`` load ``ModelSpec.instantiate`` performs leaves it a mapping;
    both reach ``model_coefficients``, so both must give the same coefficients.
    """
    validated: Any = TurbulencePropertiesConfig.load(case_dir=_dict_path(dict_name))
    assert model_coefficients(validated, model, coeffs_type).model_dump() == expected
