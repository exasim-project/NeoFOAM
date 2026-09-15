# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Wire ``incompressibleFluid`` configs to/from JSON-schema form values.

Bridges the form data the case notebooks collect (or the AI agent produces) and
the per-field split the wizard uses — a field config's ``internalField`` lives
on the *Initial values* tab while its ``boundaryField`` lives on the *BCs* tab.

BC / field values no longer need a write-boundary normalisation pass: the
:data:`neofoam.fields.value_types.FieldValue` type serialises itself to the
right per-writer form (raw for forms/JSON, ``uniform``/``nonuniform`` literals
for OpenFOAM via the writer's serialization context).

Pure pydantic + stdlib. See :mod:`neofoam.io.pydantic_schema` for the generic
(config-agnostic) schema/form helpers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = [
    "INPUT_KEYS",
    "field_name",
    "is_scheme_config",
    "merge_field_config",
    "split_field_dump",
]

#: Top-level field-config properties that belong on the *Initial values* tab
#: (the complement, ``boundaryField``, belongs on the *BCs* tab).
INPUT_KEYS = ("dimensions", "internalField")


def is_scheme_config(cls: type) -> bool:
    """True if ``cls`` writes a ``system/fv*`` file (fvSchemes / fvSolution)."""
    io = getattr(cls, "io_config", None)
    return bool(io is not None and io.file.startswith("system/fv"))


def field_name(cls: type) -> str:
    """The short field name for a ``0/<name>`` field config (e.g. ``"U"``)."""
    io = cls.io_config  # type: ignore[attr-defined]
    name: str = io.file.split("/")[-1]
    return name


def split_field_dump(
    dump: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a field config dump into (initial-values half, BCs half).

    Mirrors how the wizard renders a field across two tabs: the input half
    carries :data:`INPUT_KEYS` (``dimensions`` / ``internalField``), the BC half
    carries ``boundaryField``.
    """
    input_half = {k: dump[k] for k in INPUT_KEYS if k in dump}
    bc_half = {"boundaryField": dump.get("boundaryField", {})}
    return input_half, bc_half


def merge_field_config(
    cls: type[BaseModel],
    input_data: dict[str, Any] | None,
    bc_data: dict[str, Any] | None,
) -> BaseModel:
    """Rebuild a field config from its two form halves.

    Starts from the config's ``model_construct`` defaults so a half the user
    never submitted falls back rather than vanishing, overlays the input and BC
    halves, then validates. The OpenFOAM literal form is produced at write time
    by the :data:`~neofoam.fields.value_types.FieldValue` serializer.
    """
    data = cls.model_construct().model_dump(by_alias=True, exclude_none=True)
    data.update(input_data or {})
    data.update(bc_data or {})
    return cls.model_validate(data)
