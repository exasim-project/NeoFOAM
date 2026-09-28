# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the declarative front door: a case's system/setFields.yaml.

The specs are checked-in case files (``cases/*/system/setFields.yaml``) — exactly
what a user writes — because reading them *is* the behaviour under test. Two
properties carry the design: a region is an open mapping resolved against the
``Node`` registry, so the errors have to name the position and the offending
``type`` rather than surface a pydantic union error, and a value's shape is what
picks the field type, so a value that is neither a scalar nor a 3-vector is
refused at the file. No OpenFOAM — nothing is read or written here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neofoam.postprocess import Binary, Box, Sphere
from neofoam.preprocess import SetFieldsConfig, resolve_regions
from neofoam.preprocess.config import load_config, spec_file

CASES = Path(__file__).parent / "cases"


def _config(name: str) -> SetFieldsConfig:
    return load_config(CASES / name)


def test_a_declared_default_and_region_resolve_to_a_selector_and_its_values() -> None:
    regions = resolve_regions(_config("declared"))

    assert _config("declared").defaults == {"alpha.water": 0.0}
    assert regions[0] == (Box(min=(0, 0, -1), max=(0.1461, 0.292, 1)), {"alpha.water": 1.0})


def test_a_nested_region_resolves_through_the_node_union() -> None:
    selector, _ = resolve_regions(_config("declared"))[1]

    assert selector == Binary(
        op="or",
        left=Sphere(center=(0, 0, 0), radius=0.25),
        right=Box(min=(0.35, -1, 0), max=(1.2, 1, 0.124)),
    )


def test_a_vector_value_is_read_as_a_three_tuple() -> None:
    _, values = resolve_regions(_config("declared"))[1]

    assert values == {"alpha.water": 1.0, "U": (0.5, 0.0, 0.0)}


def test_an_unknown_region_type_names_the_position_and_the_type() -> None:
    with pytest.raises(ValueError, match=r"regions\[0\]: cannot resolve region 'sausage'"):
        resolve_regions(_config("unknown_region"))


def test_a_node_that_is_not_a_selector_is_refused() -> None:
    with pytest.raises(ValueError, match=r"regions\[0\]: cannot resolve region 'mean'"):
        resolve_regions(_config("not_a_selector"))


def test_the_error_lists_the_registered_selector_types() -> None:
    with pytest.raises(ValueError, match=r"registered selector types: .*'box'.*'sphere'"):
        resolve_regions(_config("unknown_region"))


@pytest.mark.parametrize("case", ["typo_key", "typo_region_key", "bad_value"])
def test_a_file_the_schema_does_not_accept_is_refused(case: str) -> None:
    with pytest.raises(ValidationError):
        _config(case)


def test_a_case_without_a_spec_file_declares_nothing(tmp_path: Path) -> None:
    assert spec_file(tmp_path) is None
    assert load_config(tmp_path) == SetFieldsConfig()
