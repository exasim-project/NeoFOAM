# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Tests for aliased-key IO patterns.

Demonstrates:
- A ``Field(alias="div(phi,U)")`` whose alias is not a valid Python identifier
  round-trips under the alias on disk, not the sanitised Python attribute.

The OpenFOAM scheme dictionaries (and other OpenFOAM section keys like
``snGrad(p_rgh)``) are full of non-identifier keys; ``Pimple_fvSchemes``'s
synthesised inner sections register every entry as a ``Field(alias=...)`` —
see ``src/neofoam/foam/fv_configs.py``.
"""

import pytest

from pydantic import ConfigDict, Field
from neofoam.io import (
    BaseConfig,
    YAML,
    JSON,
    OF,
    IOStrategy,
)


@IOStrategy(YAML("aliased.yaml"))
class AliasedYAMLConfig(BaseConfig):
    model_config = ConfigDict(populate_by_name=True)
    scheme_for_phi_u: str = Field(alias="div(phi,U)")


@IOStrategy(JSON("aliased.json"))
class AliasedJSONConfig(BaseConfig):
    model_config = ConfigDict(populate_by_name=True)
    scheme_for_phi_u: str = Field(alias="div(phi,U)")


@IOStrategy(OF("aliased.of"))
class AliasedOpenFOAMConfig(BaseConfig):
    model_config = ConfigDict(populate_by_name=True)
    scheme_for_phi_u: str = Field(alias="div(phi,U)")


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (AliasedYAMLConfig, "aliased.yaml"),
        (AliasedJSONConfig, "aliased.json"),
        (AliasedOpenFOAMConfig, "aliased.of"),
    ],
)
def test_aliased_field_lands_under_alias(tmp_path, config_class, filename):
    """The on-disk key is the alias, not the sanitised Python attribute name."""
    cfg = config_class.model_validate({"div(phi,U)": "Gauss upwind"})
    cfg.save(case_dir=tmp_path, file=filename)

    raw = (tmp_path / filename).read_text()
    assert "div(phi,U)" in raw
    assert "scheme_for_phi_u" not in raw


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (AliasedYAMLConfig, "aliased.yaml"),
        (AliasedJSONConfig, "aliased.json"),
        (AliasedOpenFOAMConfig, "aliased.of"),
    ],
)
def test_aliased_field_round_trip(tmp_path, config_class, filename):
    """Reloading recovers the value under the Python attribute."""
    cfg = config_class.model_validate({"div(phi,U)": "Gauss upwind"})
    cfg.save(case_dir=tmp_path, file=filename)

    reloaded = config_class.load(case_dir=tmp_path, file=filename)
    assert reloaded.scheme_for_phi_u == "Gauss upwind"
