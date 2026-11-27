# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tag a numerics form whose model declares fields for the per-field layout (headless, pure)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from neofoam.linear_solvers import FOAM_TO_GINKGO_SOLVERS, registered_preconditioners
from neofoam.ui.form_schema import (
    NF_ARM_LABEL,
    NF_BACKEND,
    NF_FIELD,
    NF_FIELD_FORM,
    NF_FIELD_GROUPS,
    NF_FINAL_OF,
    NF_REFUSED_ARMS,
    NF_SWITCH_DROPS,
)

_SOLVERS = "solvers"
_DISCRIMINATOR = "nfDiscriminator"
_CONFIG_FILE_KEY = "configFile"
_CONFIG_FILE_LABEL = "Config file"
_BACKEND_NAMES = {"openfoam": "OpenFOAM"}
_PROBE_VALUES: dict[str, Any] = {"string": "x", "number": 1.0, "integer": 1}


def _declares_fields(section: Any) -> bool:
    """True when a form section holds an entry assigned to a field."""
    properties = section.get("properties", {}) if isinstance(section, dict) else {}
    return any(NF_FIELD in entry for entry in properties.values())


def _field_groups(section: dict[str, Any]) -> dict[str, list[str]]:
    """The section's entries grouped by the field they belong to."""
    groups: dict[str, list[str]] = {}
    for key, entry in section["properties"].items():
        if NF_FIELD in entry:
            groups.setdefault(entry[NF_FIELD], []).append(key)
    return groups


def _arm_probe(arm: dict[str, Any], discriminator: str) -> dict[str, Any]:
    """The smallest block selecting ``arm``, with a sample for each required key."""
    properties = arm["properties"]
    required = {
        key: _PROBE_VALUES[properties[key]["type"]]
        for key in arm.get("required", [])
        if properties[key].get("type") in _PROBE_VALUES
    }
    return {**required, discriminator: properties[discriminator]["const"]}


def _accepts(adapter: TypeAdapter[Any], block: dict[str, Any]) -> bool:
    """Whether the field's own validators accept ``block``."""
    try:
        adapter.validate_python(block)
    except ValidationError:
        return False
    return True


def _refused_alone(
    adapter: TypeAdapter[Any], arm: dict[str, Any], probe: dict[str, Any], shared: set[str]
) -> list[str]:
    """The keys shared with other arms that this arm refuses when set on their own."""
    return [
        key
        for key, entry in arm["properties"].items()
        if key in shared
        and isinstance(entry.get("type"), str)
        and entry["type"] in _PROBE_VALUES
        and not _accepts(adapter, {**probe, key: _PROBE_VALUES[entry["type"]]})
    ]


def _tag_block_arms(block: dict[str, Any], adapter: TypeAdapter[Any]) -> None:
    """Mark the arms a field refuses and the values a switch to an arm must drop."""
    discriminator = block[_DISCRIMINATOR]
    arms = block["oneOf"]
    refused = []
    for arm in arms:
        probe = _arm_probe(arm, discriminator)
        if not _accepts(adapter, probe):
            refused.append(arm["properties"][discriminator]["const"])
            continue
        others = [other for other in arms if other is not arm]
        shared = {key for other in others for key in other["properties"]} - {discriminator}
        drops = _refused_alone(adapter, arm, probe, shared)
        if drops:
            arm[NF_SWITCH_DROPS] = drops
        if _CONFIG_FILE_KEY in arm["properties"]:
            arm[NF_ARM_LABEL] = _CONFIG_FILE_LABEL
    if refused:
        block[NF_REFUSED_ARMS] = refused


def _tag_solvers(section: dict[str, Any], cls: type[BaseModel]) -> None:
    """Add the field groups and each typed block's refused choices to ``solvers``."""
    section[NF_FIELD_GROUPS] = _field_groups(section)
    section[NF_FINAL_OF] = {
        key: entry[NF_FINAL_OF]
        for key, entry in section["properties"].items()
        if NF_FINAL_OF in entry
    }
    model = cls.model_fields[_SOLVERS].annotation
    fields = {
        info.alias or name: info
        for name, info in model.model_fields.items()  # type: ignore[union-attr]
    }
    for key, block in section["properties"].items():
        if NF_FIELD in block and _DISCRIMINATOR in block:
            _tag_block_arms(block, TypeAdapter(fields[key].rebuild_annotation()))


def _backend_note(cls: type[BaseModel]) -> dict[str, Any] | None:
    """The linear-solver backend the model fixes and the OpenFOAM words it reads."""
    backend = getattr(cls, "_backend", None)
    if not isinstance(backend, str):
        return None
    name = _BACKEND_NAMES.get(backend, backend.capitalize())
    if backend == "openfoam":
        return {"name": name, "openfoam": []}
    words = [plugin.mapper_word for plugin in registered_preconditioners(backend)]
    openfoam = [
        *FOAM_TO_GINKGO_SOLVERS,
        *(word for word in words if word not in FOAM_TO_GINKGO_SOLVERS),
    ]
    return {"name": name, "openfoam": [word for word in openfoam if word != "none"]}


def tag_field_form(schema: dict[str, Any], cls: type[BaseModel]) -> dict[str, Any]:
    """Add the per-field layout keywords when ``cls`` assigns its entries to fields.

    Use it on a finished :func:`~neofoam.ui.form_schema.jsonforms_schema`. A schema whose
    entries carry no field is returned as the same object, so such a form keeps its
    generic layout; any other gets a tagged copy. Stateless and thread-safe.

    Example::

        schema = tag_field_form(jsonforms_schema(raw), config_class)
    """
    sections = schema.get("properties", {})
    if not any(_declares_fields(section) for section in sections.values()):
        return schema
    schema = {**deepcopy(schema), NF_FIELD_FORM: True}
    sections = schema["properties"]
    if _declares_fields(sections.get(_SOLVERS)):
        _tag_solvers(sections[_SOLVERS], cls)
        note = _backend_note(cls)
        if note is not None:
            schema[NF_BACKEND] = note
    return schema
