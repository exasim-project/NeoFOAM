# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Tests for extensible IO patterns (``ConfigDict(extra="allow")``).

Demonstrates:
- Undeclared keys carried by ``extra="allow"`` round-trip through write +
  reload, with values preserved on the reloaded instance.

The synthesised fvSchemes/fvSolution section models in
``src/neofoam/foam/fv_configs.py`` use ``extra="allow"`` so OpenFOAM tutorial
keys not declared on the schema (notably the ``default`` entry on every
``ddtSchemes``/``divSchemes``/… section) can still be written and loaded.
"""

import pytest

from pydantic import ConfigDict
from neofoam.io import (
    BaseConfig,
    YAML,
    JSON,
    OF,
    IOStrategy,
)


@IOStrategy(YAML("extensible.yaml"))
class ExtensibleYAMLConfig(BaseConfig):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    declared: str


@IOStrategy(JSON("extensible.json"))
class ExtensibleJSONConfig(BaseConfig):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    declared: str


@IOStrategy(OF("extensible.of"))
class ExtensibleOpenFOAMConfig(BaseConfig):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    declared: str


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (ExtensibleYAMLConfig, "extensible.yaml"),
        (ExtensibleJSONConfig, "extensible.json"),
        (ExtensibleOpenFOAMConfig, "extensible.of"),
    ],
)
def test_extras_persist_on_write(tmp_path, config_class, filename):
    """An undeclared key kept by ``extra="allow"`` lands on disk."""
    cfg = config_class.model_validate({"declared": "a", "undeclared": "b"})
    cfg.save(case_dir=tmp_path, file=filename)

    raw = (tmp_path / filename).read_text()
    assert "undeclared" in raw
    assert "b" in raw


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (ExtensibleYAMLConfig, "extensible.yaml"),
        (ExtensibleJSONConfig, "extensible.json"),
        (ExtensibleOpenFOAMConfig, "extensible.of"),
    ],
)
def test_extras_round_trip(tmp_path, config_class, filename):
    """Reloading recovers the undeclared key on the instance via ``extra="allow"``."""
    cfg = config_class.model_validate({"declared": "a", "undeclared": "b"})
    cfg.save(case_dir=tmp_path, file=filename)

    reloaded = config_class.load(case_dir=tmp_path, file=filename)
    assert reloaded.declared == "a"
    assert getattr(reloaded, "undeclared", None) == "b"
