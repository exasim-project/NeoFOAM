# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Unit tests for ConfigContext and is_configurable_field."""

import pytest
from pydantic.fields import FieldInfo

from neofoam.framework.initialization.config_context import (
    ConfigContext,
    is_configurable_field,
)


# --- ConfigContext: register / get ---


def test_register_then_get_roundtrip(config: ConfigContext) -> None:
    """Round-trip: register a model and retrieve it."""
    obj = {"viscosity": 0.01}
    config.register("transport", obj)
    assert config.get("transport") is obj


def test_get_unknown_returns_none(config: ConfigContext) -> None:
    """Getting an unregistered name returns None."""
    assert config.get("nonexistent") is None


def test_contains_true(config: ConfigContext) -> None:
    """contains() returns True for registered names."""
    config.register("algo", "alg")
    assert config.contains("algo") is True


def test_contains_false(config: ConfigContext) -> None:
    """contains() returns False for unknown names."""
    assert config.contains("unknown") is False


def test_all_returns_copy(config: ConfigContext) -> None:
    """all() returns a copy of the current region's models."""
    config.register("a", 1)
    config.register("b", 2)
    result = config.all()
    assert result == {"a": 1, "b": 2}
    # Must be a copy — mutating it shouldn't affect the context
    result["c"] = 3
    assert config.get("c") is None


# --- Multi-region ---


def test_multi_region_register_then_get_roundtrip(config: ConfigContext) -> None:
    """Register in different regions and access via dot notation."""
    config.register("temperature", 300.0, region="fluid")
    config.register("temperature", 400.0, region="solid")

    assert config.get("fluid.temperature") == 300.0
    assert config.get("solid.temperature") == 400.0


def test_contains_cross_region(config: ConfigContext) -> None:
    """contains() works with region.name paths."""
    config.register("temp", 1, region="fluid")
    assert config.contains("fluid.temp") is True
    assert config.contains("solid.temp") is False


def test_regions_property(config: ConfigContext) -> None:
    """regions lists all registered region names."""
    config.register("a", 1, region="fluid")
    config.register("b", 2, region="solid")
    assert set(config.regions) == {"default", "fluid", "solid"}


def test_all_specific_region(config: ConfigContext) -> None:
    """all(region) returns models from that region only."""
    config.register("x", 10, region="other")
    config.register("y", 20, region="other")
    assert config.all(region="other") == {"x": 10, "y": 20}
    assert config.all() == {}  # default region is empty


# --- get_by_type / get_by_prefix ---


class TransportModel:
    pass


class TurbulenceModel:
    pass


def test_get_by_type(config: ConfigContext) -> None:
    """get_by_type filters registered models by class."""
    t1 = TransportModel()
    t2 = TurbulenceModel()
    config.register("transport", t1)
    config.register("turbulence", t2)

    result = config.get_by_type(TransportModel)
    assert result == [t1]


def test_get_by_type_empty(config: ConfigContext) -> None:
    """get_by_type returns empty list when no match."""
    config.register("something", "not_a_transport")
    assert config.get_by_type(TransportModel) == []


def test_get_by_prefix(config: ConfigContext) -> None:
    """get_by_prefix returns models whose names start with a prefix."""
    config.register("heat_source_1", "hs1")
    config.register("heat_source_2", "hs2")
    config.register("pressure", "p")

    result = config.get_by_prefix("heat_source_")
    assert result == {"heat_source_1": "hs1", "heat_source_2": "hs2"}


def test_get_by_prefix_empty(config: ConfigContext) -> None:
    """get_by_prefix returns empty dict when no match."""
    config.register("pressure", "p")
    assert config.get_by_prefix("heat_") == {}


# --- __getattr__ ---


def test_getattr_access(config: ConfigContext) -> None:
    """Attribute-style access works like get()."""
    config.register("algorithm", "PIMPLE")
    assert config.algorithm == "PIMPLE"


def test_getattr_missing_raises(config: ConfigContext) -> None:
    """Accessing unregistered name via attribute raises AttributeError."""
    with pytest.raises(AttributeError, match="No model registered"):
        _ = config.nonexistent


def test_getattr_private_raises(config: ConfigContext) -> None:
    """Accessing _-prefixed names raises AttributeError (avoids recursion)."""
    with pytest.raises(AttributeError):
        _ = config._something


def test_hasattr_registered(config: ConfigContext) -> None:
    """hasattr() returns True for registered models (via __getattr__)."""
    config.register("algo", "x")
    assert hasattr(config, "algo") is True


def test_hasattr_unregistered(config: ConfigContext) -> None:
    """hasattr() returns False for unregistered names."""
    assert hasattr(config, "nope") is False


# --- is_configurable_field ---


def _configurable_field_info() -> FieldInfo:
    field_info = FieldInfo(annotation=str)
    field_info.metadata = ["configurable"]
    return field_info


@pytest.mark.parametrize(
    "field_info,expected",
    [
        pytest.param(
            _configurable_field_info(),
            True,
            id="with-configurable-metadata",
        ),
        pytest.param(FieldInfo(), False, id="regular-field"),
        pytest.param(type("NoMeta", (), {})(), False, id="no-metadata"),
    ],
)
def test_is_configurable_field(field_info: object, expected: bool) -> None:
    """is_configurable_field handles configurable, regular, and no-metadata cases."""
    assert is_configurable_field(field_info) is expected


# --- get_configurable_fields ---


def test_get_configurable_fields_no_model(config: ConfigContext) -> None:
    """Returns empty dict when model not found."""
    assert config.get_configurable_fields("missing") == {}


def test_get_configurable_fields_non_pydantic(config: ConfigContext) -> None:
    """Returns empty dict for non-Pydantic models."""
    config.register("plain", {"key": "value"})
    assert config.get_configurable_fields("plain") == {}


# --- _parse_path disambiguation ---


def test_dotted_name_without_region(config: ConfigContext) -> None:
    """Dotted names where the prefix is NOT a region are treated as literals."""
    config.register("models.transport", "transport_obj")
    # 'models' is not a registered region, so the whole string is the key
    assert config.get("models.transport") == "transport_obj"
    assert config.contains("models.transport") is True


def test_dotted_name_with_known_region(config: ConfigContext) -> None:
    """Dotted names where the prefix IS a region are split on the dot."""
    config.register("temperature", 300.0, region="fluid")
    # 'fluid' IS a registered region
    assert config.get("fluid.temperature") == 300.0
