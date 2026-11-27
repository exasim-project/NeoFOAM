# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The OpenFOAM backend: native solver and preconditioner names, kept as written."""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, ClassVar, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    WithJsonSchema,
    field_validator,
    model_serializer,
    model_validator,
)

from neofoam.core.field_metadata import advanced
from neofoam.linear_solvers.base import (
    _FILE_READERS,
    _ROLE_RULES,
    _ROLE_SCHEMAS,
    LinearSolver,
    Preconditioner,
    _plugin_tag,
    _PreconditionedSolver,
    _registered_plugins,
    registered_preconditioners,
)

# Valid names per matrix type, from OpenFOAM v2406's own selection-table error lists.
_FOAM_CHOICES: dict[str, dict[str, frozenset[str]]] = {
    "pressure": {
        "solver": frozenset({"FPCG", "GAMG", "PBiCGStab", "PCG", "PPCG", "PPCR", "smoothSolver"}),
        "preconditioner": frozenset({"DIC", "FDIC", "GAMG", "diagonal", "distributedDIC", "none"}),
        "smoother": frozenset(
            {"DIC", "DICGaussSeidel", "FDIC", "GaussSeidel", "nonBlockingGaussSeidel"}
            | {"symGaussSeidel"}
        ),
    },
    "momentum": {
        "solver": frozenset({"GAMG", "PBiCG", "PBiCGStab", "smoothSolver"}),
        "preconditioner": frozenset({"DILU", "GAMG", "diagonal", "distributedDILU", "none"}),
        "smoother": frozenset(
            {"DILU", "DILUGaussSeidel", "GaussSeidel", "nonBlockingGaussSeidel", "symGaussSeidel"}
        ),
    },
}
_FOAM_SMOOTHERS = sorted(
    _FOAM_CHOICES["pressure"]["smoother"] | _FOAM_CHOICES["momentum"]["smoother"]
)
_FOAM_SYMMETRIC_ONLY_SOLVERS = frozenset({"PCG", "PPCG", "PPCR", "FPCG"})
_FOAM_OTHER = "Other"


def _foam_refused_choices(role: str) -> dict[str, list[str]]:
    """The native names OpenFOAM refuses for this field's matrix type."""
    other = "momentum" if role == "pressure" else "pressure"
    return {
        kind: sorted(_FOAM_CHOICES[other][kind] - valid)
        for kind, valid in _FOAM_CHOICES[role].items()
    }


def _is_file_dump(info: SerializationInfo) -> bool:
    """True when dumping for the OpenFOAM writer rather than the form."""
    return isinstance(info.context, dict) and info.context.get("format") == "openfoam"


class _KeepsFileOrder(BaseModel):
    """Remembers the order of the keys it was read with, so a save keeps it."""

    _keys: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="wrap")
    @classmethod
    def _remember_keys(cls, data: Any, handler: Any) -> Any:
        """Record the input's keys."""
        model = handler(data)
        if isinstance(data, dict):
            model._keys = tuple(data)
        return model

    def _in_read_order(self, data: dict[str, Any]) -> dict[str, Any]:
        """``data`` with the keys read from the file first, in their order."""
        return {**{key: data[key] for key in self._keys if key in data}, **data}


class OpenFoamPreconditioner(_KeepsFileOrder, Preconditioner):
    """Base of OpenFOAM preconditioners; ``type`` is the name OpenFOAM selects.

    Written as a bare word without options, else as ``preconditioner { preconditioner X; ... }``.
    Keys it does not model are kept. Subclass and decorate with ``@Preconditioner.register``.

    Example::

        @Preconditioner.register
        class FoamMine(OpenFoamPreconditioner):
            mapper_word: ClassVar[str] = "mine"
            type: Literal["mine"] = "mine"
    """

    model_config = ConfigDict(extra="allow")

    backend: ClassVar[str] = "openfoam"

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any, info: SerializationInfo) -> Any:
        """The form's object, or OpenFOAM's word or dictionary in a file."""
        data = {key: value for key, value in handler(self).items() if value is not None}
        data = self._in_read_order(data)
        if not _is_file_dump(info):
            return data
        if set(data) == {"type"}:
            return self.type
        return {"preconditioner" if key == "type" else key: value for key, value in data.items()}


def _symmetric_only(solver: str) -> Optional[str]:
    """Why a symmetric-only preconditioner cannot run with ``solver``."""
    return "needs a symmetric matrix" if solver == "PBiCG" else None


def _asymmetric_only(solver: str) -> Optional[str]:
    """Why an asymmetric-only preconditioner cannot run with ``solver``."""
    return "needs an asymmetric matrix" if solver in _FOAM_SYMMETRIC_ONLY_SOLVERS else None


@Preconditioner.register
class FoamDiagonal(OpenFoamPreconditioner):
    """OpenFOAM diagonal preconditioner."""

    mapper_word: ClassVar[str] = "diagonal"
    type: Literal["diagonal"] = "diagonal"


@Preconditioner.register
class FoamDIC(OpenFoamPreconditioner):
    """Diagonal incomplete Cholesky, for symmetric matrices."""

    mapper_word: ClassVar[str] = "DIC"
    type: Literal["DIC"] = "DIC"

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        """Symmetric matrices only."""
        return _symmetric_only(solver)


@Preconditioner.register
class FoamFDIC(OpenFoamPreconditioner):
    """Faster diagonal incomplete Cholesky, for symmetric matrices."""

    mapper_word: ClassVar[str] = "FDIC"
    type: Literal["FDIC"] = "FDIC"

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        """Symmetric matrices only."""
        return _symmetric_only(solver)


@Preconditioner.register
class FoamDistributedDIC(OpenFoamPreconditioner):
    """Distributed diagonal incomplete Cholesky, for symmetric matrices."""

    mapper_word: ClassVar[str] = "distributedDIC"
    type: Literal["distributedDIC"] = "distributedDIC"

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        """Symmetric matrices only."""
        return _symmetric_only(solver)


@Preconditioner.register
class FoamDILU(OpenFoamPreconditioner):
    """Diagonal incomplete LU, for asymmetric matrices."""

    mapper_word: ClassVar[str] = "DILU"
    type: Literal["DILU"] = "DILU"

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        """Asymmetric matrices only."""
        return _asymmetric_only(solver)


@Preconditioner.register
class FoamDistributedDILU(OpenFoamPreconditioner):
    """Distributed diagonal incomplete LU, for asymmetric matrices."""

    mapper_word: ClassVar[str] = "distributedDILU"
    type: Literal["distributedDILU"] = "distributedDILU"

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        """Asymmetric matrices only."""
        return _asymmetric_only(solver)


class _GamgOptions(BaseModel):
    """The controls ``GAMGSolver`` and its agglomeration read, with OpenFOAM's defaults."""

    nCellsInCoarsestLevel: Optional[int] = advanced(None, group="Agglomeration", default_hint=10)
    agglomerator: Optional[str] = advanced(None, group="Agglomeration", default_hint="faceAreaPair")
    mergeLevels: Optional[int] = advanced(None, group="Agglomeration")
    cacheAgglomeration: Optional[str] = advanced(None, group="Agglomeration", default_hint="true")
    nPreSweeps: Optional[int] = advanced(None, group="Sweeps", default_hint=0)
    preSweepsLevelMultiplier: Optional[int] = advanced(None, group="Sweeps", default_hint=1)
    maxPreSweeps: Optional[int] = advanced(None, group="Sweeps", default_hint=4)
    nPostSweeps: Optional[int] = advanced(None, group="Sweeps", default_hint=2)
    postSweepsLevelMultiplier: Optional[int] = advanced(None, group="Sweeps", default_hint=1)
    maxPostSweeps: Optional[int] = advanced(None, group="Sweeps", default_hint=4)
    nFinestSweeps: Optional[int] = advanced(None, group="Sweeps", default_hint=2)
    interpolateCorrection: Optional[str] = advanced(None, group="Correction", default_hint="false")
    scaleCorrection: Optional[str] = advanced(None, group="Correction")
    directSolveCoarsest: Optional[str] = advanced(None, group="Correction", default_hint="false")


@Preconditioner.register
class FoamGAMGPreconditioner(_GamgOptions, OpenFoamPreconditioner):
    """GAMG cycles as a preconditioner, nested in the solver's ``preconditioner`` block."""

    mapper_word: ClassVar[str] = "GAMG"
    type: Literal["GAMG"] = "GAMG"
    smoother: Optional[str] = advanced(None, group="Smoothing", examples=_FOAM_SMOOTHERS)
    tolerance: Optional[float] = advanced(None, group="Convergence")
    relTol: Optional[float] = advanced(None, group="Convergence")
    maxIter: Optional[int] = advanced(None, group="Convergence")
    minIter: Optional[int] = advanced(None, group="Convergence")
    nVcycles: Optional[int] = advanced(None, group="Convergence", default_hint=2)


def _foam_preconditioner_object(value: Any) -> Any:
    """The form's object for OpenFOAM's word or ``preconditioner { preconditioner X; }``."""
    if isinstance(value, str):
        return {"type": value}
    if isinstance(value, dict) and "preconditioner" in value:
        return {"type" if key == "preconditioner" else key: item for key, item in value.items()}
    return value


class OpenFoamSolver(_KeepsFileOrder, LinearSolver):
    """Base of OpenFOAM solvers, named as OpenFOAM selects them; unmodelled keys are kept.

    Only the keys read from the file or the form are written, in their order. Subclass it
    and decorate with ``@LinearSolver.register``.

    Example::

        @LinearSolver.register
        class FoamMine(OpenFoamSolver):
            solver: Literal["mine"] = "mine"
    """

    model_config = ConfigDict(extra="allow")

    backend: ClassVar[str] = "openfoam"

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any, info: SerializationInfo) -> dict[str, Any]:  # type: ignore[override]
        """The set keys in the order they were read."""
        written = {"solver", *self.model_fields_set, *(self.model_extra or {})}
        data = {
            key: value
            for key, value in handler(self).items()
            if key in written and value is not None
        }
        return self._file_block(self._in_read_order(data), info)

    def _file_block(self, data: dict[str, Any], info: SerializationInfo) -> dict[str, Any]:
        """The dumped keys as this block writes them."""
        return data


class _FoamKrylovSolver(OpenFoamSolver, _PreconditionedSolver):
    """A Krylov solver; a missing ``preconditioner`` stays missing in the file."""

    # Pydantic merges configs base by base, so the second base's ``forbid`` would win.
    model_config = ConfigDict(extra="allow")

    # Typed as the backend's preconditioner union by _retype_solver. The default is for the
    # form: ``_dump`` writes only keys that were set.
    preconditioner: Any = Field({"type": "none"}, validate_default=True)

    @field_validator("preconditioner", mode="before")
    @classmethod
    def _word_to_object(cls, value: Any) -> Any:
        """Read OpenFOAM's word or nested dictionary as the object form."""
        return _foam_preconditioner_object(value)


@LinearSolver.register
class FoamPCG(_FoamKrylovSolver):
    """Preconditioned conjugate gradient, for symmetric matrices."""

    solver: Literal["PCG"] = "PCG"


@LinearSolver.register
class FoamPPCG(_FoamKrylovSolver):
    """Pipelined preconditioned conjugate gradient, for symmetric matrices."""

    solver: Literal["PPCG"] = "PPCG"


@LinearSolver.register
class FoamPPCR(_FoamKrylovSolver):
    """Pipelined preconditioned conjugate residual, for symmetric matrices."""

    solver: Literal["PPCR"] = "PPCR"


@LinearSolver.register
class FoamFPCG(_FoamKrylovSolver):
    """Faster preconditioned conjugate gradient, for symmetric matrices."""

    solver: Literal["FPCG"] = "FPCG"


@LinearSolver.register
class FoamPBiCG(_FoamKrylovSolver):
    """Preconditioned bi-conjugate gradient, for asymmetric matrices."""

    solver: Literal["PBiCG"] = "PBiCG"


@LinearSolver.register
class FoamPBiCGStab(_FoamKrylovSolver):
    """Stabilized preconditioned bi-conjugate gradient, for either matrix type."""

    solver: Literal["PBiCGStab"] = "PBiCGStab"


@LinearSolver.register
class FoamSmoothSolver(OpenFoamSolver):
    """Iterative solver built on a smoother."""

    solver: Literal["smoothSolver"] = "smoothSolver"
    smoother: Optional[str] = Field(None, examples=_FOAM_SMOOTHERS)
    nSweeps: Optional[int] = advanced(None, default_hint=1)


@LinearSolver.register
class FoamGAMG(_GamgOptions, OpenFoamSolver):
    """Geometric-algebraic multigrid solver."""

    solver: Literal["GAMG"] = "GAMG"
    smoother: Optional[str] = Field(None, examples=_FOAM_SMOOTHERS)


# A coupled solver sets a control per component, as ``(1e-05 1e-05 1e-05)``.
_ScalarOrComponents = Annotated[
    Union[float, str, None],
    Field(union_mode="left_to_right"),
    WithJsonSchema({"type": ["number", "string"]}),
]


@LinearSolver.register
class FoamOtherBlock(OpenFoamSolver):
    """A block the typed solvers cannot represent, kept exactly as read.

    Holds library solvers, unknown names and choices invalid for the field; OpenFOAM
    remains the judge of them at run time. ``solverName`` is the file's ``solver`` word.

    Example::

        FoamOtherBlock.model_validate({"solver": "Other", "solverName": "petsc"})
    """

    model_config = ConfigDict(title="Other (kept as written)")

    solver: Literal["Other"] = "Other"
    solverName: Optional[str] = None
    tolerance: _ScalarOrComponents = None  # type: ignore[assignment]
    relTol: _ScalarOrComponents = None  # type: ignore[assignment]

    def _file_block(self, data: dict[str, Any], info: SerializationInfo) -> dict[str, Any]:
        """Put the file's own solver word back."""
        if not _is_file_dump(info):
            return data
        return {
            "solver" if key == "solverName" else key: value
            for key, value in data.items()
            if key != "solver"
        }


def _foam_other_block(block: dict[str, Any]) -> dict[str, Any]:
    """``block`` as the ``Other`` arm, with its solver word in place."""
    if "solver" not in block:
        return {"solver": _FOAM_OTHER, **block}
    renamed = {"solverName" if key == "solver" else key: item for key, item in block.items()}
    return {"solver": _FOAM_OTHER, **renamed}


def _foam_preconditioner_word(block: dict[str, Any]) -> Any:
    """The name a block's ``preconditioner`` selects, or ``None``."""
    selected = _foam_preconditioner_object(block.get("preconditioner"))
    return selected.get("type") if isinstance(selected, dict) else None


def _foam_block_is_typed(block: dict[str, Any]) -> bool:
    """True when a typed OpenFOAM plugin can hold this block."""
    plugins = {_plugin_tag(plugin): plugin for plugin in _registered_plugins("openfoam")}
    solver = plugins.get(block.get("solver"))  # type: ignore[arg-type]
    if solver is None or isinstance(block.get("smoother"), dict):
        return False
    if "preconditioner" not in solver.model_fields or "preconditioner" not in block:
        return True
    word = _foam_preconditioner_word(block)
    known = {_plugin_tag(p, "type"): p for p in registered_preconditioners("openfoam")}
    if word not in known or isinstance(_nested_smoother(block), dict):
        return False
    return known[word].incompatibility(_plugin_tag(solver)) is None


def _nested_smoother(block: dict[str, Any]) -> Any:
    """The ``smoother`` inside a dictionary-form preconditioner."""
    value = block.get("preconditioner")
    return value.get("smoother") if isinstance(value, dict) else None


def _read_foam_block(block: Any) -> Any:
    """Keep a block the typed plugins cannot represent as the ``Other`` arm."""
    if not isinstance(block, dict) or block.get("solver") == _FOAM_OTHER:
        return block
    return block if _foam_block_is_typed(block) else _foam_other_block(block)


def _foam_block_for_role(block: Any, role: str) -> Any:
    """Keep a block whose names OpenFOAM refuses for this field's matrix as ``Other``."""
    if not isinstance(block, dict) or block.get("solver") == _FOAM_OTHER:
        return block
    choices = _FOAM_CHOICES[role]
    named = [("solver", block.get("solver")), ("smoother", block.get("smoother"))]
    if "preconditioner" in block and _plugin_accepts_preconditioner(block):
        named += [
            ("preconditioner", _foam_preconditioner_word(block)),
            ("smoother", _nested_smoother(block)),
        ]
    refused = any(isinstance(word, str) and word not in choices[kind] for kind, word in named)
    return _foam_other_block(block) if refused else block


def _plugin_accepts_preconditioner(block: dict[str, Any]) -> bool:
    """True when the block's solver reads a ``preconditioner`` entry."""
    plugins = {_plugin_tag(plugin): plugin for plugin in _registered_plugins("openfoam")}
    plugin = plugins.get(block.get("solver"))  # type: ignore[arg-type]
    return plugin is not None and "preconditioner" in plugin.model_fields


def _foam_role_schema(role: str) -> dict[str, Any]:
    """The wizard keywords of an OpenFOAM field."""
    return {"nfRefusedChoices": _foam_refused_choices(role)}


_FILE_READERS["openfoam"] = _read_foam_block
_ROLE_SCHEMAS["openfoam"] = _foam_role_schema
for _role in _FOAM_CHOICES:
    _ROLE_RULES[("openfoam", _role)] = partial(_foam_block_for_role, role=_role)
