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

from pydantic import ConfigDict, Field, create_model, model_validator
from pydantic.fields import FieldInfo

from neofoam.foam.schemes import (
    Corrected,
    DdtScheme,
    DivScheme,
    Euler,
    GaussDiv,
    GaussGrad,
    GaussLaplacian,
    GradScheme,
    InterpolationScheme,
    LaplacianScheme,
    Linear,
    SnGradScheme,
    Upwind,
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


def _expand_default(cls: type, data: Any) -> Any:
    """Fill missing required entries from an OpenFOAM ``default`` shorthand.

    A scheme section may name every operator explicitly *or* give a single
    ``default`` (``gradSchemes { default Gauss linear; }``) that OpenFOAM applies
    to every operator it doesn't spell out. The typed per-operator fields are
    required, so a tutorial that leans on ``default`` would fail to load. This
    before-validator expands ``default`` into any declared field the input omits
    (explicit entries always win), so real cases round-trip while validation still
    proves each operator the solver needs is covered.
    """
    if not isinstance(data, dict) or "default" not in data:
        return data
    default_value = data["default"]
    # ``default none`` is OpenFOAM's *sentinel* — "no default; an unlisted operator
    # is an error" — not a value to fill with. Expanding it would fabricate an
    # invalid ``none`` scheme; leave the required keys missing so the real gap shows.
    if isinstance(default_value, str) and default_value.strip() == "none":
        return data
    out = dict(data)
    for name, field_info in cls.model_fields.items():  # type: ignore[attr-defined]
        alias = field_info.alias or name
        if alias == "default":
            continue
        if alias not in out and name not in out and field_info.is_required():
            out[alias] = default_value
    return out


# The decorated before-validator, built once and attached to every synthesized
# section model. Typed ``Any``: create_model's ``__validators__`` wants decorated
# validators, whose pydantic descriptor type isn't expressible at the call site.
_EXPAND_DEFAULT_VALIDATOR: Any = model_validator(mode="before")(
    classmethod(_expand_default)  # type: ignore[arg-type]
)


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
            # Attach the ``default``-expansion before-validator to the synthesized
            # section model so a tutorial that leans on ``default`` round-trips.
            __validators__={"_expand_default": _EXPAND_DEFAULT_VALIDATOR},
            **field_defs,
        )
        cls.model_fields[section_name] = FieldInfo(  # type: ignore[attr-defined]
            annotation=fresh, default=None
        )
    cls.model_rebuild(force=True)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Canonical starter schemes/solvers (the form prefill — see ``form_defaults``)
# ---------------------------------------------------------------------------


def _canonical_scheme(section_name: str, alias: str) -> Optional[dict[str, Any]]:
    """A runnable default value for one scheme entry, serialized as a form value.

    One canonical scheme per section (``Gauss linear`` grad/div, ``corrected``
    snGrad, …). The one nuance: a ``div`` term over a *flux* (``div(phi,U)``) gets a
    bounded ``upwind`` interpolation, while the viscous-stress divergence stays
    ``linear`` — the split OpenFOAM tutorials always make. Returns ``None`` for an
    unrecognised (passthrough ``str``) section.
    """
    if section_name == "ddtSchemes":
        return Euler().model_dump(by_alias=True)
    if section_name == "gradSchemes":
        return GaussGrad(interpolation=Linear()).model_dump(by_alias=True)
    if section_name == "divSchemes":
        interp = Upwind() if "phi" in alias else Linear()
        return GaussDiv(interpolation=interp).model_dump(by_alias=True)
    if section_name == "laplacianSchemes":
        return GaussLaplacian(interpolation=Linear(), sn_grad=Corrected()).model_dump(
            by_alias=True
        )
    if section_name == "snGradSchemes":
        return Corrected().model_dump(by_alias=True)
    if section_name == "interpolationSchemes":
        return Linear().model_dump(by_alias=True)
    return None


#: Standard corrector counts per algorithm-control section (keyed by section name).
_CONTROL_CORRECTORS: dict[str, dict[str, Any]] = {
    "PIMPLE": {"nOuterCorrectors": 1, "nCorrectors": 2, "nNonOrthogonalCorrectors": 0},
    "PISO": {"nCorrectors": 2, "nNonOrthogonalCorrectors": 0},
    "SIMPLE": {"nNonOrthogonalCorrectors": 0},
}


def _canonical_controls(
    section_name: str, declared_aliases: list[str]
) -> dict[str, Any]:
    """A runnable algorithm-control block (e.g. ``PIMPLE { … }``) for the scaffold.

    The corrector counts come from the section name; the algorithm reads the block at
    run time even though the schema models it as optional/extra, so a starter must
    include it. Any declared control key (the closed-domain pressure reference
    ``pRefCell`` / ``pRefValue``) is seeded to 0 — harmless when a pressure BC already
    references the field, required when none does.
    """
    block: dict[str, Any] = dict(_CONTROL_CORRECTORS.get(section_name, {}))
    for alias in declared_aliases:
        block[alias] = 0
    return block


def _canonical_solver(alias: str) -> dict[str, Any]:
    """A runnable linear-solver block for one ``solvers.<field>`` entry.

    Pressure-like fields (``p`` / ``p_rgh`` / ``pcorr`` / ``Phi``) get a symmetric
    ``PCG``/``DIC`` block; everything else (``U``, ``alpha.water``, …) a
    ``smoothSolver``. A ``<field>Final`` companion tightens ``relTol`` to 0.
    """
    is_final = alias.endswith("Final")
    base = alias[: -len("Final")] if is_final else alias
    b = base.lower()
    is_pressure = b == "p" or b.startswith("p_") or b.startswith("pcorr") or b == "phi"
    if is_pressure:
        return {
            "solver": "PCG",
            "preconditioner": "DIC",
            "tolerance": 1e-06,
            "relTol": 0.0 if is_final else 0.05,
        }
    return {
        "solver": "smoothSolver",
        "smoother": "symGaussSeidel",
        "tolerance": 1e-08,
        "relTol": 0.0 if is_final else 0.1,
    }


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
    def form_defaults(cls) -> Optional[dict[str, object]]:
        """A canonical, ready-to-edit prefill for every required scheme this spec needs.

        The per-operator scheme fields are required-but-defaultless (so validation
        proves each operator is covered), which leaves ``model_construct`` — and thus
        the form prefill — empty. This fills each required entry with a runnable
        canonical scheme (:func:`_canonical_scheme`), keyed by its OpenFOAM alias.
        """
        out: dict[str, Any] = {}
        for section_name, entries in cls._pending_sections.items():
            section_out: dict[str, Any] = {}
            for _attr, (_value_type, alias, optional) in entries.items():
                if optional:
                    continue
                scheme = _canonical_scheme(section_name, alias)
                if scheme is not None:
                    section_out[alias] = scheme
            if section_out:
                out[section_name] = section_out
        return out or None

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
    def form_defaults(cls) -> Optional[dict[str, object]]:
        """A canonical, runnable ``fvSolution`` prefill.

        Fills each required ``solvers.<field>`` entry with a runnable linear-solver
        block (:func:`_canonical_solver`), and emits an algorithm-control block for
        each declared control section (``PIMPLE`` / ``PISO`` / ``SIMPLE`` —
        :func:`_canonical_controls`) so the case actually runs: the solver reads e.g.
        ``PIMPLE`` at run time even though the schema treats it as optional/extra.
        """
        out: dict[str, Any] = {}
        solvers = cls._pending_sections.get("solvers", {})
        solvers_out: dict[str, Any] = {}
        for _attr, (_value_type, alias, optional) in solvers.items():
            if optional:
                continue
            solvers_out[alias] = _canonical_solver(alias)
        if solvers_out:
            out["solvers"] = solvers_out
        for section_name, entries in cls._pending_sections.items():
            if section_name == "solvers":
                continue
            aliases = [alias for _attr, (_vt, alias, _opt) in entries.items()]
            block = _canonical_controls(section_name, aliases)
            if block:
                out[section_name] = block
        return out or None

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
