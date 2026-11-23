# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared blocks, plugins and helpers of the linear-solver specs.

Blocks live on disk in ``test/foam/cases/neonLinearSolvers/system`` so they go through the
real OpenFOAM parser. Late-registration helpers undo their plugin, as the registry is global.
"""

import shutil
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

import pybFoam as pyf

from neofoam.core.plugin_system import PluginSystem
from neofoam.foam import (
    GinkgoPreconditioner,
    GinkgoSolver,
    LinearSolver,
    LinearSolverConversionWarning,
    Preconditioner,
    fvSolution,
    linear_solver_entry,
    neonFvSolution,
    refresh_linear_solver_slices,
)
from neofoam.framework.model import Model
from neofoam.io.strategies.openfoam_strategy import _to_python

_CASE = Path(__file__).parents[1] / "foam" / "cases" / "neonLinearSolvers"
_FILE = {"format": "openfoam"}

_SIMPLE_BLOCKS = [
    ("CgIc", "Cg"),
    ("CgNone", "Cg"),
    ("CgMultigrid", "Cg"),
    ("CgMultigridTuned", "Cg"),
    ("CgMultigridAdvanced", "Cg"),
    ("GmresKrylov", "Gmres"),
    ("BicgstabJacobiBlocks", "Bicgstab"),
    ("IrRelaxed", "Ir"),
    ("BicgIlu", "Bicg"),
    ("BicgstabJacobi", "Bicgstab"),
    ("GmresIc", "Gmres"),
    ("IrJacobi", "Ir"),
    ("IrIlu", "Ir"),
    ("ConfigFile", "GinkgoConfigFile"),
    ("ConfigFileControls", "GinkgoConfigFile"),
]

_LEGACY_BLOCKS = [
    (
        "PCG",
        {
            "solver": "Cg",
            "preconditioner": {"type": "Ic"},
            "tolerance": 1e-06,
            "relTol": 0.05,
            "maxIter": 100,
        },
    ),
    ("PCGL1On", {"solver": "Cg", "preconditioner": {"type": "Jacobi"}}),
    (
        "PCGWithGAMG",
        {
            "solver": "Cg",
            "preconditioner": {"type": "Multigrid", "nCellsInCoarsestLevel": 10, "nPreSweeps": 2},
        },
    ),
    ("PBiCG", {"solver": "Bicg", "preconditioner": {"type": "Ilu"}, "tolerance": 1e-06}),
    ("PBiCGStab", {"solver": "Bicgstab", "preconditioner": {"type": "Jacobi"}, "relTol": 0.1}),
    ("smoothSolver", {"solver": "Ir", "preconditioner": {"type": "Jacobi"}, "tolerance": 1e-08}),
    ("smoothSolverDILU", {"solver": "Ir", "preconditioner": {"type": "Ilu"}}),
    (
        "GAMG",
        {
            "solver": "Cg",
            "preconditioner": {"type": "Multigrid", "nCellsInCoarsestLevel": 10, "nPreSweeps": 2},
            "tolerance": 1e-06,
        },
    ),
    ("GAMGReuse", {"solver": "Cg", "preconditioner": {"type": "Multigrid"}}),
    (
        "GAMGAdvancedKeys",
        {"solver": "Cg", "preconditioner": {"type": "Multigrid", "nCellsInCoarsestLevel": 10}},
    ),
    ("NativeCg", {"solver": "Cg", "preconditioner": {"type": "Jacobi"}, "tolerance": 1e-06}),
    ("NativeIc", {"solver": "Cg", "preconditioner": {"type": "Ic"}}),
    ("NativeWord", {"solver": "Bicg", "preconditioner": {"type": "Ilu"}, "maxIter": 77}),
    (
        "NativeCriteria",
        {
            "solver": "Bicgstab",
            "preconditioner": {"type": "Ilu"},
            "tolerance": 1e-06,
            "relTol": 0.01,
        },
    ),
    ("NativeGmres", {"solver": "Gmres", "preconditioner": {"type": "none"}}),
    ("NativeIrDefault", {"solver": "Ir", "preconditioner": {"type": "Jacobi"}, "tolerance": 1e-06}),
]

_WARNED_KEYS = [
    ("smoothSolver", ["nSweeps", "smoother"]),
    ("smoothSolverDILU", ["smoother"]),
    ("GAMG", ["agglomerator", "mergeLevels", "nPostSweeps", "smoother"]),
    ("GAMGReuse", ["reuseSetup", "smoother"]),
    ("GAMGAdvancedKeys", ["cycle", "max_levels", "nCoarsestSweeps", "smoother"]),
]

_SILENT_LEGACY = ["PCG", "PCGL1On", "PCGWithGAMG", "PBiCG", "NativeCg", "NativeCriteria"]

_REJECTED = [
    "ConfigFileWithType",
    "ConfigFileWithTolerance",
    "ConfigFileNoSolver",
    "ConfigFileAbsolute",
    "ConfigFileOpenFOAMSolver",
    "StrayType",
    "GinkgoNoType",
    "LowerCaseGinkgo",
    "TopLevelMultigrid",
    "UnknownGinkgoType",
    "NoSolver",
    "AlreadyMapped",
    "CgUnknownKey",
    "CgBackendKey",
    "CgCriteriaIteration",
    "CgCriteriaRelative",
    "IrRelaxation",
    "PCGUnknownKey",
    "PCGBackendKey",
    "GAMGBackendKey",
    "GAMGCoupled",
    "SmoothSolverMules",
    "MultigridOnBicg",
    "MultigridOnBicgstab",
    "MultigridOnGmres",
    "MultigridOnIr",
    "PBiCGWithGAMG",
    "PBiCGStabWithGAMG",
    "IrNone",
    "SmoothSolverNone",
    "PCGNoPreconditioner",
    "NativeNoPreconditioner",
    "PCGFdic",
    "NativeSchwarz",
    "NativeTunedIc",
    "L1Off",
    "CheckFrequency",
    "ReuseOff",
    "MinIterFactor",
    "TuningWithoutMultigrid",
    "SimpleUnknownKey",
    "UnknownPreconditioner",
    "TuningOnIc",
    "TunedMultigridOnBicg",
    "UnknownPreconditionerOption",
]


def _blocks(name: str) -> dict[str, Any]:
    """Every block of one fixture file, as the OpenFOAM parser returns them."""
    return _to_python(pyf.dictionary.read(str(_CASE / "system" / name)))


def _entry() -> Any:
    """The entry type of the only backend NeoN solvers declare."""
    return linear_solver_entry("ginkgo")


def _validated(block: dict[str, Any]) -> Any:
    """Validate a block, ignoring conversion warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinearSolverConversionWarning)
        return _entry().model_validate(block)


def _slice(name: str, base: type[fvSolution] = neonFvSolution) -> Any:
    """A per-spec slice declaring ``p`` and ``U``, built the way operations build it."""
    slice_cls = Model(name).config(base)
    slice_cls.add("p", "U")
    return slice_cls


def _case_with_p(tmp_path: Path, block: dict[str, Any]) -> Path:
    """A case copy whose ``solvers.p`` is ``block``."""
    case = tmp_path / "case"
    shutil.copytree(_CASE, case)
    path = case / "system" / "fvSolution"
    root = pyf.dictionary.read(str(path))
    solvers = root.subDict("solvers")
    solvers.remove("p")
    solvers.subDictOrAdd("p")
    _fill(solvers.subDict("p"), block)
    root.write(str(path))
    return case


def _fill(node: Any, block: dict[str, Any]) -> None:
    """Copy a parsed block into a pybFoam dictionary."""
    for key, value in block.items():
        if isinstance(value, dict):
            node.subDictOrAdd(key)
            _fill(node.subDict(key), value)
        else:
            node.set(key, value)


class _Fgmres(GinkgoSolver):
    """A solver registered after import."""

    solver: Literal["Fgmres"] = "Fgmres"


class _OtherBackend(LinearSolver):
    """A solver of a backend no NeoN slice declares."""

    backend: ClassVar[str] = "amgx"
    solver: Literal["AmgxCg"] = "AmgxCg"


class _DummyPreconditioner(GinkgoPreconditioner):
    """A preconditioner registered after the slices are built."""

    mapper_word: ClassVar[str] = "diagonal"
    type: Literal["Dummy"] = "Dummy"


class _CgOnlyPreconditioner(GinkgoPreconditioner):
    """A late preconditioner that declares a solver restriction."""

    mapper_word: ClassVar[str] = "diagonal"
    type: Literal["CgOnly"] = "CgOnly"

    @classmethod
    def incompatibility(cls, solver: str) -> Optional[str]:
        return None if solver == "Cg" else "Cg only"


class _OtherPreconditioner(Preconditioner):
    """A preconditioner of a backend no NeoN slice declares."""

    backend: ClassVar[str] = "amgx"
    mapper_word: ClassVar[str] = "amgxJacobi"
    type: Literal["AmgxJacobi"] = "AmgxJacobi"


@contextmanager
def _registered_preconditioner(plugin: type[Preconditioner]) -> Iterator[None]:
    """Register ``plugin`` for the block's duration."""
    Preconditioner.register(plugin)
    try:
        yield
    finally:
        PluginSystem.remove_plugin_model("Preconditioner", plugin)
        refresh_linear_solver_slices()


@contextmanager
def _registered(plugin: type[LinearSolver]) -> Iterator[None]:
    """Register ``plugin`` for the block's duration."""
    LinearSolver.register(plugin)
    try:
        yield
    finally:
        PluginSystem.remove_plugin_model("LinearSolver", plugin)
        refresh_linear_solver_slices()
