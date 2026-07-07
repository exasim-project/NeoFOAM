# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
``fvSchemes`` / ``fvSolution`` — per-model OpenFOAM dictionary base classes.

The two classes here are :class:`BaseConfig` subclasses bound to the two
OpenFOAM case-level dictionaries. They are *not* meant to be instantiated
directly. Instead:

- ``spec.config(fvSchemes)`` returns a per-spec subclass synthesised by
  the framework (:meth:`BaseSpec.config` recognises the
  ``_synthesize_per_spec`` marker).
- ``@<Subclass>.add(div="div(phi,U)", grad="grad(U)")`` extends the
  subclass with typed Pydantic fields whose value types come from
  :mod:`neofoam.foam.schemes` (``DdtScheme``, ``DivScheme``, …).
- ``@<Subclass>.add(...)`` doubles as an operation decorator so it can
  stack on ``@spec.operation(...)``. The decorator is a no-op at call
  time — its only effect is the side effect of extending the class.

Loading reads the underlying OpenFOAM file once per case dir; slicing
is per-subclass and per-section.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional

from pydantic import ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from neofoam.foam.schemes import (
    DdtScheme,
    DivScheme,
    GradScheme,
    InterpolationScheme,
    LaplacianScheme,
    SnGradScheme,
)
from neofoam.io import BaseConfig, IOStrategy, OF


# ---------------------------------------------------------------------------
# Short-name → (section, value type) lookup tables
# ---------------------------------------------------------------------------

_SCHEMES_SECTIONS: dict[str, tuple[str, Any]] = {
    "ddt": ("ddtSchemes", DdtScheme),
    "div": ("divSchemes", DivScheme),
    "grad": ("gradSchemes", GradScheme),
    "laplacian": ("laplacianSchemes", LaplacianScheme),
    "snGrad": ("snGradSchemes", SnGradScheme),
    "interpolation": ("interpolationSchemes", InterpolationScheme),
}


def _sanitize_name(s: str) -> str:
    """OpenFOAM key (``"div(phi,U)"``) → Python attr (``"div_phi_U"``).

    The Pydantic field uses the original key as its ``alias`` so OpenFOAM
    dictionary parsing finds the entry; ``populate_by_name=True`` also
    lets callers use the sanitized name directly when constructing
    instances in Python.
    """
    return (
        s.replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "_")
        .replace(" ", "_")
        .replace("*", "_")
    )


def _register_entry(
    cls: type,
    section_name: str,
    key: str,
    value_type: Any,
    optional: bool = False,
) -> None:
    """Record one (section, key, value_type, optional) on ``cls._pending_sections``.

    Storage is per-subclass; the actual Pydantic submodels are
    (re-)synthesised by :func:`_rebuild_sections` after every change.
    ``optional`` entries become ``Optional[...]`` fields defaulting to ``None``
    (for domain-dependent keys like ``pRefCell`` that not every case carries).
    """
    section = cls._pending_sections.setdefault(section_name, {})  # type: ignore[attr-defined]
    attr_name = _sanitize_name(key)
    section.setdefault(attr_name, (value_type, key, optional))


def _rebuild_sections(cls: type) -> None:
    """(Re)synthesise every section submodel and re-attach to ``cls``.

    Pydantic v2 captures a snapshot of nested schemas at
    ``model_rebuild`` time, so adding fields to an already-attached
    section submodel doesn't propagate. Building each section from
    scratch via :func:`pydantic.create_model` and re-assigning the
    parent field works around this.
    """
    for section_name, entries in cls._pending_sections.items():  # type: ignore[attr-defined]
        field_defs: dict[str, Any] = {}
        for attr, (value_type, alias, optional) in entries.items():
            if optional:
                field_defs[attr] = (
                    Optional[value_type],
                    Field(alias=alias, default=None),
                )
            else:
                field_defs[attr] = (value_type, Field(alias=alias))
        fresh = create_model(
            f"_{section_name}",
            __config__=ConfigDict(extra="allow", populate_by_name=True),
            **field_defs,
        )
        cls.model_fields[section_name] = FieldInfo(  # type: ignore[attr-defined]
            annotation=fresh, default=None
        )
    cls.model_rebuild(force=True)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@IOStrategy(OF("system/fvSchemes"))
class fvSchemes(BaseConfig):
    """Base class for per-spec ``system/fvSchemes`` subclasses.

    Operations on a spec declare which entries they read via
    ``@<Subclass>.add(...)``. Each call extends the subclass with one
    or more typed fields, scoped to the section the short-name maps
    to (``ddt → ddtSchemes``, ``div → divSchemes``, …). Unknown short
    names pass through unchanged with ``str`` typing.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    _synthesize_per_spec: ClassVar[bool] = True
    _section_classes: ClassVar[dict[str, type]] = {}
    _pending_sections: ClassVar[dict[str, dict[str, Any]]] = {}
    _finalized: ClassVar[bool] = False

    @classmethod
    def add(
        cls, **section_to_keys: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Extend this subclass with typed entries.

        Each kwarg maps a short section name to either a single entry
        key (``"div(phi,U)"``) or a list of keys. Returns an identity
        decorator so call sites can stack ``@<Subclass>.add(...)`` on
        top of ``@spec.operation(...)``.
        """
        for short_name, keys in section_to_keys.items():
            section_name, value_type = _SCHEMES_SECTIONS.get(
                short_name, (short_name, str)
            )
            if not isinstance(keys, list):
                keys = [keys]
            for key in keys:
                _register_entry(cls, section_name, key, value_type)
        _rebuild_sections(cls)

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _decorator


@IOStrategy(OF("system/fvSolution"))
class fvSolution(BaseConfig):
    """Base class for per-spec ``system/fvSolution`` subclasses.

    ``@<Subclass>.add(*fields)`` declares which fields the operation
    solves for. Each field name becomes a typed entry under the
    ``solvers`` sub-dictionary; other top-level sections (``PIMPLE``,
    ``SIMPLE``, ``relaxationFactors``, …) pass through via
    ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    _synthesize_per_spec: ClassVar[bool] = True
    _pending_sections: ClassVar[dict[str, dict[str, Any]]] = {}
    _finalized: ClassVar[bool] = False

    @classmethod
    def add(cls, *fields: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Declare solver entries the operation needs.

        Each ``field`` adds a typed entry under ``solvers.<field>`` plus its
        ``solvers.<field>Final`` companion: on the final (in PISO mode:
        every) outer iteration ``fvMatrix::solve`` selects the ``Final``
        solver settings, so OpenFOAM requires both dictionaries. Today the
        value is parsed as ``dict[str, Any]`` (extra-allow sub-section); a
        richer typed solver-control model is a follow-up. The presence
        check alone catches the most common case-misconfiguration failures.
        """
        for field_name in fields:
            _register_entry(cls, "solvers", field_name, dict)
            _register_entry(cls, "solvers", f"{field_name}Final", dict)
        _rebuild_sections(cls)

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _decorator

    @classmethod
    def add_controls(cls, section: str, **key_to_type: Any) -> None:
        """Declare OPTIONAL control entries under a top-level ``section``.

        For algorithm-control keys the solver reads straight from the dict —
        e.g. ``PIMPLE { pRefCell; pRefValue; }`` for closed-domain pressure
        referencing. Optional (default ``None``) so cases that don't need them
        (open domains with a fixed-pressure BC) still validate, and
        ``exclude_none`` keeps them out of the written file unless set.
        """
        for key, value_type in key_to_type.items():
            _register_entry(cls, section, key, value_type, optional=True)
        _rebuild_sections(cls)
