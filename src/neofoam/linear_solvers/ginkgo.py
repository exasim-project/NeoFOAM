# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The Ginkgo backend: its solvers, preconditioners and ``configFile`` fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_serializer,
    model_validator,
)

from neofoam.core.field_metadata import advanced
from neofoam.linear_solvers.base import (
    _CONTROL_KEYS,
    _FILE_READERS,
    _ROLE_RULES,
    _ROLE_SCHEMAS,
    LinearSolver,
    Preconditioner,
    _PreconditionedSolver,
    _without_null_defaults,
)
from neofoam.linear_solvers.ginkgo_reader import (
    _TRUE_WORDS,
    _converted_pressure_block,
    _read_ginkgo_block,
)


def _reject_vector_gmres(entry: RootModel[Any]) -> RootModel[Any]:
    """Reject Gmres on a vector field: a zero component's solve returns NaN with no error."""
    if entry.root.solver == "Gmres":
        raise ValueError(
            "Gmres cannot solve U: an all-zero component (any 2D case) gives NaN; "
            "use Bicgstab, Bicg or Ir"
        )
    return entry


class GinkgoPreconditioner(Preconditioner):
    """Base of Ginkgo preconditioners; ``mapper_word`` is the name the NeoN mapper expands.

    Subclass it and decorate with ``@Preconditioner.register``. Override
    ``incompatibility`` to refuse a solver. ``native_block`` is the full Ginkgo dictionary
    an existing file may spell it as.

    Example::

        @Preconditioner.register
        class Sor(GinkgoPreconditioner):
            mapper_word: ClassVar[str] = "SOR"
            type: Literal["Sor"] = "Sor"
    """

    backend: ClassVar[str] = "ginkgo"


@Preconditioner.register
class NoPreconditioner(GinkgoPreconditioner):
    """No preconditioning."""

    model_config = ConfigDict(title="none")

    shared_backends: ClassVar[tuple[str, ...]] = ("openfoam",)
    mapper_word: ClassVar[str] = "none"

    type: Literal["none"] = "none"

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        """Iterative refinement needs an inner solver: without one it diverges silently."""
        return "unpreconditioned Ir diverges" if solver == "Ir" else None


@Preconditioner.register
class Jacobi(GinkgoPreconditioner):
    """Point Jacobi."""

    mapper_word: ClassVar[str] = "diagonal"
    native_block: ClassVar[Optional[dict[str, Any]]] = {
        "type": "preconditioner::Jacobi",
        "max_block_size": "1",
    }

    type: Literal["Jacobi"] = "Jacobi"
    max_block_size: Optional[int] = advanced(None, default_hint=1)


@Preconditioner.register
class Ic(GinkgoPreconditioner):
    """Incomplete Cholesky."""

    mapper_word: ClassVar[str] = "DIC"
    native_block: ClassVar[Optional[dict[str, Any]]] = {
        "type": "preconditioner::Ic",
        "factorization": {"type": "factorization::ParIc"},
    }

    type: Literal["Ic"] = "Ic"


@Preconditioner.register
class Ilu(GinkgoPreconditioner):
    """Incomplete LU."""

    mapper_word: ClassVar[str] = "DILU"
    native_block: ClassVar[Optional[dict[str, Any]]] = {
        "type": "preconditioner::Ilu",
        "factorization": {"type": "factorization::ParIlu"},
    }

    type: Literal["Ilu"] = "Ilu"


class PgmCoarsening(BaseModel):
    """Pgm coarsening options of :class:`Multigrid`; unset ones stay out of the file."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: Optional[int] = advanced(None, default_hint=15)
    max_unassigned_ratio: Optional[float] = advanced(None, default_hint=0.05)
    deterministic: Optional[bool] = advanced(None, default_hint=False)
    skip_sorting: Optional[bool] = advanced(None, default_hint=False)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        """Drop ``default: null``, which JSONForms would write into the data."""
        return _without_null_defaults(handler(core_schema), handler)

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any) -> dict[str, Any]:
        """Dump the set keys."""
        return {key: value for key, value in handler(self).items() if value is not None}


@Preconditioner.register
class Multigrid(GinkgoPreconditioner):
    """Algebraic multigrid; the mapper drops unknown keys on this path."""

    mapper_word: ClassVar[str] = "GAMG"
    drops_unknown_keys: ClassVar[bool] = True
    foam_options: ClassVar[tuple[str, ...]] = ("nCellsInCoarsestLevel", "nPreSweeps")

    type: Literal["Multigrid"] = "Multigrid"
    nCellsInCoarsestLevel: Optional[int] = advanced(None, group="Hierarchy", default_hint=64)
    max_levels: Optional[int] = advanced(None, group="Hierarchy", default_hint=10)
    cycle: Optional[Literal["v", "f", "w"]] = advanced(None, group="Hierarchy", default_hint="v")
    nPreSweeps: Optional[int] = advanced(None, group="Smoothing", default_hint=1)
    nCoarsestSweeps: Optional[int] = advanced(None, group="Smoothing", default_hint=8)
    coarsening: Optional[PgmCoarsening] = advanced(None, group="Pgm coarsening")

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        """PBiCG throws and PBiCGStab diverges with this multigrid."""
        return None if solver == "Cg" else "Multigrid is valid only with Cg"


class GinkgoSolver(_PreconditionedSolver):
    """Base of Ginkgo solvers, written ``solver Cg; preconditioner Ic; ...``.

    Subclass it for another Ginkgo solver; the class name in the file is its Ginkgo
    ``solver::<name>``. Combinations that fail at run time are rejected here.

    Example::

        Cg.model_validate({"solver": "Cg", "preconditioner": "Ic", "relTol": 0.05})
    """

    backend: ClassVar[str] = "ginkgo"

    # Typed as the backend's preconditioner union by _retype_solver.
    preconditioner: Any = Field({"type": "none"}, validate_default=True)

    @field_validator("preconditioner", mode="before")
    @classmethod
    def _word_to_object(cls, value: Any) -> Any:
        """Read the file's bare word as the object form."""
        return {"type": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_preconditioner(self) -> GinkgoSolver:
        """Reject a pairing the preconditioner declares unusable."""
        reason = self.preconditioner.incompatibility(self.solver)
        if reason:
            raise ValueError(f"{self.solver} with {self.preconditioner.type}: {reason}")
        return self

    def runtime_settings(self, case_dir: Path) -> dict[str, Any]:
        """The block in the shape the NeoN mapper expands, Schwarz wrapping included."""
        settings = {**self.model_dump(), "solver": "Ginkgo"}
        settings |= self.preconditioner.runtime_settings()
        return {"type": f"solver::{self.solver}", **settings}


@LinearSolver.register
class Cg(GinkgoSolver):
    """Ginkgo conjugate gradient."""

    solver: Literal["Cg"] = "Cg"


@LinearSolver.register
class Bicg(GinkgoSolver):
    """Ginkgo bi-conjugate gradient."""

    solver: Literal["Bicg"] = "Bicg"


@LinearSolver.register
class Bicgstab(GinkgoSolver):
    """Ginkgo stabilized bi-conjugate gradient."""

    solver: Literal["Bicgstab"] = "Bicgstab"


@LinearSolver.register
class Gmres(GinkgoSolver):
    """Ginkgo restarted GMRES."""

    solver: Literal["Gmres"] = "Gmres"
    krylov_dim: Optional[int] = advanced(None, default_hint=100)


@LinearSolver.register
class Ir(GinkgoSolver):
    """Ginkgo iterative refinement."""

    solver: Literal["Ir"] = "Ir"
    preconditioner: Any = Field({"type": "Jacobi"}, validate_default=True)
    relaxation_factor: Optional[float] = advanced(None, default_hint=1)


_CONFIG_FILE_TEXT = (
    "Fallback for Ginkgo settings the simple solvers do not cover. "
    "The JSON file is not read or checked here."
)


# Registered last so the fallback is never the form's first arm or default.
@LinearSolver.register
class GinkgoConfigFile(LinearSolver):
    """Fallback for Ginkgo settings the simple solvers lack: a JSON file path.

    Written ``solver Ginkgo; configFile x;``. The JSON is neither parsed nor validated;
    a relative path is resolved against the case directory at solver start. NeoN applies
    the numeric controls here only with ``l1ScaledResidual`` on, so they are rejected
    without it.

    Example::

        GinkgoConfigFile.model_validate({"solver": "Ginkgo", "configFile": "cg.json"})
    """

    model_config = ConfigDict(
        title="Ginkgo config file (fallback)",
        json_schema_extra={"description": _CONFIG_FILE_TEXT},
    )

    backend: ClassVar[str] = "ginkgo"

    solver: Literal["Ginkgo"] = "Ginkgo"
    configFile: str = Field(title="JSON file (relative to the case directory)")
    l1ScaledResidual: Optional[str] = None
    checkFrequency: Optional[int] = None
    minIterFactor: Optional[float] = None
    reuseSetup: Optional[str] = None

    @model_validator(mode="after")
    def _check_block(self) -> GinkgoConfigFile:
        """Reject a path the runtime cannot open and controls it would ignore."""
        if not self.configFile or self.configFile.startswith("/") or '"' in self.configFile:
            raise ValueError(f"configFile needs a relative, unquoted path; got {self.configFile!r}")
        controls = [key for key in _CONTROL_KEYS if getattr(self, key) is not None]
        if controls and str(self.l1ScaledResidual).lower() not in _TRUE_WORDS:
            raise ValueError(f"{controls} apply to a configFile only with l1ScaledResidual on")
        return self

    def runtime_settings(self, case_dir: Path) -> dict[str, Any]:
        """The block with its JSON path anchored at the case directory."""
        return {**self.model_dump(), "configFile": str(case_dir / self.configFile)}


def _ginkgo_role_schema(role: str) -> dict[str, Any]:
    """The wizard keywords of a Ginkgo field with a role rule."""
    return {"nfDefaultChoices": {"preconditioner": "Jacobi"}}


_FILE_READERS["ginkgo"] = _read_ginkgo_block
_ROLE_RULES[("ginkgo", "pressure")] = _converted_pressure_block
_ROLE_SCHEMAS["ginkgo"] = _ginkgo_role_schema
