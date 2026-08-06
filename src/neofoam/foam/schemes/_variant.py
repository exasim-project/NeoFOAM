# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared serialisation gate for the scheme variant models.

A scheme variant has two faces: the OpenFOAM token a case file carries (``Euler``,
``Gauss upwind``, ``limited 0.33``) and the structured object its pydantic model and
JSON Schema describe. Emitting the token *unconditionally* made ``model_dump()``
contradict ``model_json_schema()`` — and the case wizard hands both to JSONForms at
once, so a value like ``"Gauss upwind"`` matched no ``oneOf`` arm, the combinator
renderer fell back to arm 0, and the form displayed ``none`` for a case that says
``Gauss upwind``. Wrong values on screen, silently.

So the token form is opt-in, gated on ``context={"format": "openfoam"}``. That is the
mechanism :class:`neofoam.fields.value_types.FieldValue` and
:class:`neofoam.tools.block_mesh.BlockMeshDictConfig` already use, and which both
OpenFOAM write paths already pass (``io.write_configs``, ``io.dictfile``) — so file
output is unaffected and only schema-shaped consumers see the change.

A variant that composes another (``Gauss <interpolation>``) must pass
:data:`OPENFOAM_CONTEXT` down when dumping the inner scheme, or it would splice a dict
into its own token.
"""

from typing import Any

from pydantic import BaseModel, SerializationInfo, model_serializer

__all__ = ["OPENFOAM_CONTEXT", "SchemeVariant"]

#: Dump context selecting OpenFOAM token output over the structured shape.
OPENFOAM_CONTEXT: dict[str, Any] = {"format": "openfoam"}


class SchemeVariant(BaseModel):
    """Structured data by default; an OpenFOAM token under ``OPENFOAM_CONTEXT``."""

    def openfoam_str(self) -> str:
        """Return the OpenFOAM token for this variant, e.g. ``Gauss upwind``."""
        raise NotImplementedError

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any, info: SerializationInfo) -> Any:
        if (info.context or {}).get("format") != "openfoam":
            return handler(self)
        return self.openfoam_str()
