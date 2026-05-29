# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the viscosityModel plugin interface and native registration."""

from typing import Union, get_args, get_origin

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelSpec
from neofoam.viscosity.interface import Model, viscosityModel

# Importing the models package registers the bundled natives (Newtonian).
import neofoam.viscosity.models  # noqa: F401


def test_viscosity_interface_registered_in_plugin_system() -> None:
    assert PluginSystem.get_registered("viscosityModel") is not None


def test_newtonian_registered_with_interface() -> None:
    assert "Newtonian" in viscosityModel.registered_names()


def test_all_specs_returns_model_spec_objects() -> None:
    specs = viscosityModel.all_specs()
    assert specs, "expected at least the Newtonian spec"
    assert all(isinstance(spec, ModelSpec) for spec in specs)
    assert "Newtonian" in [spec.name for spec in specs]


def test_find_spec_returns_spec_by_name() -> None:
    spec = viscosityModel.find_spec("Newtonian")
    assert isinstance(spec, ModelSpec)
    assert spec.name == "Newtonian"


def test_find_spec_unknown_returns_none() -> None:
    assert viscosityModel.find_spec("CrossPowerLaw") is None


def test_newtonian_detect_is_true() -> None:
    spec = viscosityModel.find_spec("Newtonian")
    assert spec is not None
    assert spec.run_detect() is True


def test_plugin_model_is_discriminated_union(clean_viscosity_registry: None) -> None:
    """The interface exposes a pydantic model whose ``model`` field is a
    discriminated union over every registered viscosity model."""
    Model("CrossPowerLaw").register_with(viscosityModel)
    Model("BirdCarreau").register_with(viscosityModel)

    plugin_model = viscosityModel.plugin_model  # type: ignore[attr-defined]
    assert isinstance(plugin_model, type)
    assert issubclass(plugin_model, BaseModel)

    field = plugin_model.model_fields["model"]
    assert get_origin(field.annotation) is Union
    assert field.discriminator == "model_type"

    member_types = {
        get_args(member.model_fields["model_type"].annotation)[0]
        for member in get_args(field.annotation)
    }
    assert {"Newtonian", "CrossPowerLaw", "BirdCarreau"} <= member_types

    discriminator = plugin_model.model_json_schema()["properties"]["model"][
        "discriminator"
    ]
    assert discriminator["propertyName"] == "model_type"
    assert {"Newtonian", "CrossPowerLaw", "BirdCarreau"} <= set(
        discriminator["mapping"]
    )


def test_plugin_model_dispatches_by_discriminator(
    clean_viscosity_registry: None,
) -> None:
    """Validation selects the right variant purely from the discriminator."""
    Model("CrossPowerLaw").register_with(viscosityModel)

    instance = viscosityModel.create(  # type: ignore[attr-defined]
        model={"model_type": "CrossPowerLaw"}
    )

    assert instance.model.model_type == "CrossPowerLaw"
