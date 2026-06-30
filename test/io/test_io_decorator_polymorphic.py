# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Tests for polymorphic (discriminated-union) IO patterns.

Demonstrates:
- A nested ``BaseModel`` whose ``@model_serializer`` flattens it to a flat
  on-disk entry (a string like ``"Gauss linear"``) round-trips through the
  paired ``BeforeValidator`` on read.

This is the schema pattern every ``src/neofoam/foam/schemes/*.py`` scheme
uses (``DdtScheme``, ``DivScheme``, …) to map between the rich pydantic
discriminated-union form and the terse OpenFOAM dictionary entry form.
"""

from typing import Annotated, Any, Literal

import pytest

from pydantic import BaseModel, BeforeValidator, model_serializer

from neofoam.io import (
    BaseConfig,
    YAML,
    JSON,
    OF,
    IOStrategy,
)


class FlatScheme(BaseModel):
    """Stand-in for a discriminated-union scheme variant.

    ``@model_serializer`` flattens to ``"Gauss <interp>"`` (the OpenFOAM
    dictionary entry form), and the paired :func:`_parse_scheme` parses
    it back. Real schemes use the same pair plus a ``Discriminator("type")``;
    one variant is enough to exercise the round-trip.
    """

    type: Literal["Gauss"] = "Gauss"
    interpolation: str

    @model_serializer
    def serialize(self) -> str:
        return f"Gauss {self.interpolation}"


def _parse_scheme(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    parts = v.split(maxsplit=1)
    return {"type": parts[0], "interpolation": parts[1] if len(parts) > 1 else ""}


SchemeRef = Annotated[FlatScheme, BeforeValidator(_parse_scheme)]


@IOStrategy(YAML("polymorphic.yaml"))
class PolymorphicYAMLConfig(BaseConfig):
    laplacian: SchemeRef


@IOStrategy(JSON("polymorphic.json"))
class PolymorphicJSONConfig(BaseConfig):
    laplacian: SchemeRef


@IOStrategy(OF("polymorphic.of"))
class PolymorphicOpenFOAMConfig(BaseConfig):
    laplacian: SchemeRef


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (PolymorphicYAMLConfig, "polymorphic.yaml"),
        (PolymorphicJSONConfig, "polymorphic.json"),
        (PolymorphicOpenFOAMConfig, "polymorphic.of"),
    ],
)
def test_polymorphic_field_writes_flat_entry(tmp_path, config_class, filename):
    """The flat ``"Gauss linear"`` form lands on disk, not a nested sub-dict."""
    cfg = config_class(laplacian=FlatScheme(interpolation="linear"))
    cfg.save(case_dir=tmp_path, file=filename)

    raw = (tmp_path / filename).read_text()
    assert "Gauss linear" in raw


@pytest.mark.parametrize(
    "config_class,filename",
    [
        (PolymorphicYAMLConfig, "polymorphic.yaml"),
        (PolymorphicJSONConfig, "polymorphic.json"),
        (PolymorphicOpenFOAMConfig, "polymorphic.of"),
    ],
)
def test_polymorphic_field_round_trip(tmp_path, config_class, filename):
    """Reloading parses the flat entry via ``BeforeValidator`` and rebuilds the model."""
    cfg = config_class(laplacian=FlatScheme(interpolation="linear"))
    cfg.save(case_dir=tmp_path, file=filename)

    reloaded = config_class.load(case_dir=tmp_path, file=filename)
    assert reloaded.laplacian.type == "Gauss"
    assert reloaded.laplacian.interpolation == "linear"
