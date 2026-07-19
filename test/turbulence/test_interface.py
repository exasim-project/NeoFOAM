# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the momentumTransportModel plugin interface and native registration.

Expected members are derived from the *actually registered* names and from the
discovered cases, never from string literals: adding a native model (with its
case) extends these checks automatically.
"""

from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from neofoam.core.plugin_system import PluginSystem
from neofoam.framework.model import ModelSpec
from neofoam.turbulence.momentumTransport import momentumTransportModel

# Importing the package registers the bundled natives (laminar).
import neofoam.turbulence  # noqa: F401

from turbulence.conftest import CASES

NATIVE_NAMES = {
    c.selection["model_name"] for c in CASES if c.selection["resolves_to"] == "native"
}
UNREGISTERED_NAMES = {
    c.selection["model_name"]
    for c in CASES
    if c.selection["resolves_to"] == "unregistered"
}


def test_turbulence_interface_registered_in_plugin_system() -> None:
    assert PluginSystem.get_registered("momentumTransportModel") is not None


def test_native_case_models_are_registered() -> None:
    assert NATIVE_NAMES <= set(momentumTransportModel.registered_names())


def test_all_specs_returns_model_spec_objects() -> None:
    specs = momentumTransportModel.all_specs()
    assert specs, "expected at least one registered spec"
    assert all(isinstance(spec, ModelSpec) for spec in specs)


def test_find_spec_round_trips_registered_names() -> None:
    for name in momentumTransportModel.registered_names():
        spec = momentumTransportModel.find_spec(name)
        assert isinstance(spec, ModelSpec)
        assert spec.name == name


def test_find_spec_unknown_returns_none() -> None:
    # An unregistered case's model name (e.g. Smagorinsky) has no spec.
    for name in UNREGISTERED_NAMES:
        assert momentumTransportModel.find_spec(name) is None


def test_registered_specs_detect_true() -> None:
    for name in momentumTransportModel.registered_names():
        spec = momentumTransportModel.find_spec(name)
        assert spec is not None
        assert spec.run_detect() is True


def _union_members(annotation: Any) -> tuple[Any, ...]:
    # A single-member Union collapses to the bare type, so normalize both forms.
    return get_args(annotation) if get_origin(annotation) is Union else (annotation,)


def test_plugin_model_is_discriminated_union() -> None:
    """The interface exposes a pydantic model whose ``model`` field discriminates
    over every registered turbulence model by ``model_type``."""
    plugin_model = momentumTransportModel.plugin_model  # type: ignore[attr-defined]
    assert isinstance(plugin_model, type)
    assert issubclass(plugin_model, BaseModel)

    field = plugin_model.model_fields["model"]
    assert field.discriminator == "model_type"

    member_types = {
        get_args(member.model_fields["model_type"].annotation)[0]
        for member in _union_members(field.annotation)
    }
    assert set(momentumTransportModel.registered_names()) <= member_types


def test_plugin_model_dispatches_by_discriminator() -> None:
    """Validation selects the right variant purely from the discriminator."""
    name = momentumTransportModel.registered_names()[0]

    instance = momentumTransportModel.create(  # type: ignore[attr-defined]
        model={"model_type": name}
    )

    assert instance.model.model_type == name
