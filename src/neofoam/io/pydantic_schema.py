# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Helpers to drive JSON-schema forms from pydantic models.

Pure pydantic + stdlib (no marimo / ``json_schema_widget``). These back the
case notebooks (``test/agent/hotRoom/case_ui.py`` / ``case_wizard.py``), which
render each config's ``model_json_schema()`` as an ``@rjsf`` form:

- :func:`default_values` prefills a form with a config's defaults;
- :func:`rjsf_uischema` tidies the rjsf rendering of pydantic discriminated
  unions (``oneOf`` / ``anyOf`` + a ``const`` discriminator);
- :func:`slice_schema` keeps a subset of a schema's top-level properties (used
  to split a field config's form into ``internalField`` vs ``boundaryField``).
"""

from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel

from neofoam.io.base import BaseConfig

__all__ = ["default_values", "rjsf_uischema", "slice_schema"]


def default_values(cls: type[BaseModel]) -> dict[str, Any]:
    """Return a config's defaults as a form value (no validation).

    A config may supply a ready-to-edit scaffold via
    :meth:`~neofoam.io.base.BaseConfig.form_defaults` (fvSchemes/fvSolution do, whose
    fields are required-but-defaultless so ``model_construct`` alone yields ``{}``);
    otherwise the prefill comes from ``model_construct`` (partial, no validation).
    """
    if isinstance(cls, type) and issubclass(cls, BaseConfig):
        scaffold = cls.form_defaults()
        if scaffold:
            return dict(scaffold)
    try:
        return cls.model_construct().model_dump(by_alias=True, exclude_none=True)
    except Exception:
        return {}


def slice_schema(schema: dict[str, Any], keep: Iterable[str]) -> dict[str, Any]:
    """A copy of ``schema`` keeping only the named top-level properties.

    ``$defs`` is preserved untouched (referenced arms must stay reachable
    through ``$ref``); ``required`` is filtered to ``keep``.
    """
    keep_set = set(keep)
    sliced = dict(schema)
    props = schema.get("properties", {})
    sliced["properties"] = {k: v for k, v in props.items() if k in keep_set}
    if "required" in schema:
        req = [r for r in schema["required"] if r in keep_set]
        if req:
            sliced["required"] = req
        else:
            sliced.pop("required", None)
    return sliced


def rjsf_uischema(schema: dict[str, Any]) -> dict[str, Any]:
    """Tidy the ``@rjsf`` rendering of pydantic unions.

    Pydantic discriminated unions emit ``oneOf`` (schemes, with a
    ``discriminator``) or ``anyOf`` (BC unions) whose branches each carry a
    ``type`` ``const``. Out of the box rjsf renders such a field three times:
    the union dropdown, a duplicate heading for the selected branch, and an
    editable text box for the ``const`` ``type``. This builds a uiSchema that
    hides the ``const`` discriminator (``ui:widget: hidden``) — the value still
    round-trips so ``model_validate`` can pick the branch, it just isn't shown.
    The duplicate branch heading is suppressed in the widget itself (a custom
    ``TitleFieldTemplate``), so we deliberately do *not* set ``label: false``
    here (that would also hide the field/key labels like ``Ddt(U)``). A field's
    ``boundaryField`` patch dict is reached through ``additionalProperties``.
    """
    defs: dict[str, Any] = schema.get("$defs", {})

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        ref = node.get("$ref")
        if ref:
            target: dict[str, Any] = defs.get(ref.split("/")[-1], {})
            return target
        return node

    def for_node(node: dict[str, Any], seen: frozenset[Any]) -> dict[str, Any]:
        node = resolve(node)
        # Cycle guard on the (titled) ``$defs`` only — untitled wrapper nodes
        # (e.g. the ``additionalProperties`` of ``boundaryField``) share a
        # ``None`` title and must not collide with each other.
        key = node.get("title")
        if key is not None:
            if key in seen:
                return {}
            seen = seen | {key}
        ui: dict[str, Any] = {}
        union = node.get("oneOf") or node.get("anyOf")
        if union:
            # Merge each branch's tidy (hidden ``const`` ``type`` + nested unions).
            for branch in union:
                for k, v in for_object(resolve(branch), seen).items():
                    ui.setdefault(k, v)
            return ui
        if node.get("type") == "object":
            ui.update(for_object(node, seen))
            ap = node.get("additionalProperties")
            if isinstance(ap, dict):
                sub = for_node(ap, seen)
                if sub:
                    ui["additionalProperties"] = sub
        return ui

    def for_object(obj: dict[str, Any], seen: frozenset[Any]) -> dict[str, Any]:
        ui: dict[str, Any] = {}
        for name, prop in obj.get("properties", {}).items():
            pr = resolve(prop)
            if "const" in pr:
                ui[name] = {"ui:widget": "hidden"}
                continue
            sub = for_node(prop, seen)
            if sub:
                ui[name] = sub
        return ui

    return for_node(schema, frozenset())
