# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The ``system/fvSolution`` slices whose solver blocks are linear-solver plugins."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Iterator, Optional

from pydantic import AfterValidator

from neofoam.foam.fv_configs import _rebuild_sections, fvSolution
from neofoam.linear_solvers import linear_solver_entry
from neofoam.linear_solvers.base import _CONTROL_KEYS, _refresh_listeners
from neofoam.linear_solvers.ginkgo import _reject_vector_gmres

_TYPED_FIELDS = ("p", "U", "pFinal", "UFinal")
_VECTOR_FIELDS = frozenset({"U", "UFinal"})
_PRESSURE_FIELDS = ("p", "pFinal")
_FIELD_ROLES = {"p": "pressure", "pFinal": "pressure", "U": "momentum", "UFinal": "momentum"}


class _TypedFvSolution(fvSolution):
    """An ``fvSolution`` slice whose ``p``, ``U`` and ``Final`` blocks are one backend's plugins."""

    _backend: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Keep the per-spec name ``<spec>_fvSolution`` that configs are looked up by."""
        super().__init_subclass__(**kwargs)
        if cls.__name__ in ("neonFvSolution", "openfoamFvSolution"):
            return
        for base_name in ("neonFvSolution", "openfoamFvSolution"):
            cls.__name__ = cls.__name__.replace(base_name, "fvSolution")
        cls.__qualname__ = cls.__name__

    @classmethod
    def _solver_value_type(cls, field_name: str) -> Any:
        """The linear-solver entry for typed fields, else the open block."""
        if field_name in _TYPED_FIELDS:
            return linear_solver_entry(cls._backend, _FIELD_ROLES[field_name])
        return super()._solver_value_type(field_name)

    @classmethod
    def _solver_schema_extra(cls, alias: str) -> Optional[dict[str, Any]]:
        """The wizard's field group of a typed entry, and the entry a ``Final`` one mirrors."""
        if alias not in _TYPED_FIELDS:
            return None
        base = alias.removesuffix("Final")
        return {"nfField": base} if alias == base else {"nfField": base, "nfFinalOf": base}


class neonFvSolution(_TypedFvSolution):  # noqa: N801
    """``system/fvSolution`` of a NeoN solver, with ``p``, ``U`` and ``Final`` blocks typed.

    Use it through ``spec.config(neonFvSolution)``; other solvers keep :class:`fvSolution`
    and its open blocks. ``_backend`` names the only backend whose plugins the form lists.
    Turbulence fields stay open. Sections are rebuilt whenever a plugin registers.

    Example::

        Slice = spec.config(neonFvSolution)
        Slice.add("p")
    """

    _backend: ClassVar[str] = "ginkgo"

    @classmethod
    def _solver_value_type(cls, field_name: str) -> Any:
        """The typed entry, refusing Gmres on a vector field."""
        value_type = super()._solver_value_type(field_name)
        if field_name in _VECTOR_FIELDS:
            return Annotated[value_type, AfterValidator(_reject_vector_gmres)]
        return value_type

    @classmethod
    def _starter_solver(cls, alias: str) -> dict[str, Any]:
        """The starter block, in simple form for typed fields."""
        block = super()._starter_solver(alias)
        if alias not in _TYPED_FIELDS:
            return block
        controls = {key: block[key] for key in _CONTROL_KEYS if key in block}
        solver = "Cg" if block["solver"] == "PCG" else "Ir"
        return {"solver": solver, "preconditioner": {"type": "Jacobi"}, **controls}


class openfoamFvSolution(_TypedFvSolution):  # noqa: N801
    """``system/fvSolution`` of a pybFoam solver, with OpenFOAM's own solver names typed.

    Use it through ``spec.config(openfoamFvSolution)``. The typed model only describes the
    file: nothing is converted and OpenFOAM reads the file itself at run time. A block
    the plugins cannot represent loads as the ``Other`` arm and is written back as read.

    Example::

        Slice = spec.config(openfoamFvSolution)
        Slice.add("p")
    """

    _backend: ClassVar[str] = "openfoam"

    @classmethod
    def _starter_solver(cls, alias: str) -> dict[str, Any]:
        """The canonical block, with its preconditioner in the form's object shape."""
        block = super()._starter_solver(alias)
        if alias in _TYPED_FIELDS and "preconditioner" in block:
            return {**block, "preconditioner": {"type": block["preconditioner"]}}
        return block


def _slices(cls: type[_TypedFvSolution]) -> Iterator[type[_TypedFvSolution]]:
    """Every per-spec subclass of ``cls``."""
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _slices(subclass)


def _retype_slice(slice_cls: type[_TypedFvSolution]) -> None:
    """Point one slice's typed fields at the current entry type and rebuild it."""
    section = slice_cls._pending_sections.get("solvers", {})
    for attr, (_value_type, alias, optional, schema_extra) in section.items():
        if alias in _TYPED_FIELDS:
            section[attr] = (slice_cls._solver_value_type(alias), alias, optional, schema_extra)
    if section:
        _rebuild_sections(slice_cls)


def _retype_all_slices() -> None:
    """Rebuild every typed slice after a plugin change."""
    for slice_cls in _slices(_TypedFvSolution):
        _retype_slice(slice_cls)


_refresh_listeners.append(_retype_all_slices)
