# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM-specific initialization helpers."""

from pathlib import Path
from typing import Any, Optional, Type

import pybFoam as pyf

from neofoam.framework.initialization import InitStep, field, lazy, model
from neofoam.io.dictread import Value, read_toplevel


def create_arglist(argv: list[str]) -> InitStep:
    """The ``Foam::argList`` — under ``-parallel``, the MPI session itself.

    Its own step, and a *model* so the Context keeps it alive for the whole run:
    ``pyf.Time`` holds only a raw reference to the argList it was built from, and
    ``~argList`` calls ``UPstream::shutdown()`` (MPI_Finalize). Building it as a
    local inside :func:`create_runtime` ended the MPI session the moment the
    ``Time`` was constructed, and every reduction after that aborted the run with
    *MPI_Bcast called after MPI_FINALIZE*.
    """

    def create(_context: dict[str, Any]) -> Any:
        return pyf.argList(argv)

    return model("foam_arglist", create=create)


def create_runtime() -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return pyf.Time(context["models.foam_arglist"])

    return lazy("runtime", create=create, depends_on=["models.foam_arglist"])


#: Substring identifying a ``dynamicFvMesh`` that refines/unrefines cells, i.e.
#: changes mesh *topology* rather than only moving points. Matched by name because
#: that is what the case declares: every OpenFOAM refinement mesh carries it
#: (``dynamicRefineFvMesh``, ``dynamicRefineBalancedFvMesh``).
_REFINEMENT_MESH_MARKER = "Refine"


def _refinement_mesh_type(case: Path) -> Optional[str]:
    """The topology-changing ``dynamicFvMesh`` type *case* selects, else ``None``."""
    kind = read_toplevel(case / "constant" / "dynamicMeshDict", "dynamicFvMesh")
    if not isinstance(kind, Value) or _REFINEMENT_MESH_MARKER not in kind.text:
        return None
    return kind.text


def refuse_mesh_refinement() -> None:
    """Refuse a case whose ``dynamicFvMesh`` refines cells, before anything is built.

    Mesh *motion* is supported; adaptive mesh refinement is not, and is refused
    rather than run. Nothing downstream reacts to cells appearing and
    disappearing — a VoF run's MULES correction cache, ``alphaPhiUn`` and the
    isoAdvector surface would all be stale across a re-mesh — so an AMR case
    would solve on the initial static topology and produce plausible but wrong
    results.

    Call it as the *first* statement of a solver's ``mesh`` init step, ahead of
    every ``context[...]`` lookup (:func:`create_mesh` is the ready-made step):
    it reads only ``constant/dynamicMeshDict`` in the working directory, so the
    case is refused with no argList, no ``Foam::Time`` and no mesh constructed.
    """
    refinement = _refinement_mesh_type(Path("."))
    if refinement is not None:
        raise NotImplementedError(
            f"adaptive mesh refinement is not implemented: constant/dynamicMeshDict "
            f"selects dynamicFvMesh {refinement}, which changes the mesh topology "
            f"during the run. Refusing rather than solving on the initial mesh."
        )


def new_mesh(arglist: Any, runtime: Any) -> Any:
    """The case mesh: a moving ``dynamicFvMesh`` when the case asks for one.

    Mirrors ``createDynamicFvMesh.H``: a case carrying ``constant/dynamicMeshDict``
    gets the motion solver that dictionary selects, anything else keeps the plain
    static ``fvMesh`` of ``createMesh.H``. ``dynamicFvMesh::New`` would itself fall
    back to a ``staticFvMesh``, but selecting on the dictionary keeps the
    static-mesh path on exactly the code it has always run.

    Call it from a solver's own ``mesh`` init step (:func:`create_mesh` is the
    ready-made one), behind :func:`refuse_mesh_refinement`, so every solver
    selects the mesh the same way and refuses the same cases.

    The dictionary is looked up relative to the working directory, like every
    other case file this package reads.
    """
    if not Path("constant/dynamicMeshDict").is_file():
        return pyf.fvMesh(runtime)
    return pyf.dynamicFvMesh.New(arglist, runtime)


def create_mesh() -> InitStep:
    """The ``mesh`` init step — see :func:`new_mesh` for what it selects."""

    def create(context: dict[str, Any]) -> Any:
        # First, so an AMR case is refused before the argList and the Foam::Time
        # it would be built on are resolved from the context.
        refuse_mesh_refinement()
        return new_mesh(context["models.foam_arglist"], context["runtime"])

    return lazy("mesh", create=create, depends_on=["runtime", "models.foam_arglist"])


def create_time_mesh(argv: list[str]) -> list[InitStep]:
    return [create_arglist(argv), create_runtime(), create_mesh()]


def read_vol_field(field_type: Type[Any], name: str) -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return field_type.read_field(context["mesh"], name)

    return field(name, create=create, depends_on=["mesh"])
