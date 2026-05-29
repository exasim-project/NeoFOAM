# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the turbulenceModel plugin interface and native registration."""

from typing import Union, get_args, get_origin

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelSpec
from neofoam.turbulence.interface import Model, turbulenceModel

# Importing the models package registers the bundled natives (laminar).
import neofoam.turbulence.models  # noqa: F401


def test_turbulence_interface_registered_in_plugin_system() -> None:
    assert PluginSystem.get_registered("turbulenceModel") is not None


def test_laminar_registered_with_interface() -> None:
    assert "laminar" in turbulenceModel.registered_names()


def test_all_specs_returns_model_spec_objects() -> None:
    specs = turbulenceModel.all_specs()
    assert specs, "expected at least the laminar spec"
    assert all(isinstance(spec, ModelSpec) for spec in specs)
    assert "laminar" in [spec.name for spec in specs]


def test_find_spec_returns_spec_by_name() -> None:
    spec = turbulenceModel.find_spec("laminar")
    assert isinstance(spec, ModelSpec)
    assert spec.name == "laminar"


def test_find_spec_unknown_returns_none() -> None:
    assert turbulenceModel.find_spec("kEpsilon") is None


def test_laminar_detect_is_true() -> None:
    spec = turbulenceModel.find_spec("laminar")
    assert spec is not None
    assert spec.run_detect() is True


def test_plugin_model_is_discriminated_union(clean_turbulence_registry: None) -> None:
    """The interface exposes a pydantic model whose ``model`` field is a
    discriminated union over every registered turbulence model."""
    Model("kEpsilon").register_with(turbulenceModel)
    Model("kOmegaSST").register_with(turbulenceModel)

    plugin_model = turbulenceModel.plugin_model  # type: ignore[attr-defined]
    assert isinstance(plugin_model, type)
    assert issubclass(plugin_model, BaseModel)

    field = plugin_model.model_fields["model"]
    assert get_origin(field.annotation) is Union
    assert field.discriminator == "model_type"

    # Each registered model contributes one arm to the union.
    member_types = {
        get_args(member.model_fields["model_type"].annotation)[0]
        for member in get_args(field.annotation)
    }
    assert {"laminar", "kEpsilon", "kOmegaSST"} <= member_types

    # The discriminated union is reflected in the JSON schema.
    discriminator = plugin_model.model_json_schema()["properties"]["model"][
        "discriminator"
    ]
    assert discriminator["propertyName"] == "model_type"
    assert {"laminar", "kEpsilon", "kOmegaSST"} <= set(discriminator["mapping"])


def test_plugin_model_dispatches_by_discriminator(
    clean_turbulence_registry: None,
) -> None:
    """Validation selects the right variant purely from the discriminator."""
    Model("kOmegaSST").register_with(turbulenceModel)

    instance = turbulenceModel.create(  # type: ignore[attr-defined]
        model={"model_type": "kOmegaSST"}
    )

    assert instance.model.model_type == "kOmegaSST"
