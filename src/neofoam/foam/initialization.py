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

    A *model* so the Context keeps it alive for the whole run: ``pyf.Time`` only
    holds a raw reference to it, and ``~argList`` calls MPI_Finalize.
    """

    def create(_context: dict[str, Any]) -> Any:
        return pyf.argList(argv)

    return model("foam_arglist", create=create)


def create_runtime() -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return pyf.Time(context["models.foam_arglist"])

    return lazy("runtime", create=create, depends_on=["models.foam_arglist"])


#: Every OpenFOAM topology-changing (refining) ``dynamicFvMesh`` carries it in its
#: name: ``dynamicRefineFvMesh``, ``dynamicRefineBalancedFvMesh``.
_REFINEMENT_MESH_MARKER = "Refine"


def _refinement_mesh_type(case: Path) -> Optional[str]:
    """The topology-changing ``dynamicFvMesh`` type *case* selects, else ``None``."""
    kind = read_toplevel(case / "constant" / "dynamicMeshDict", "dynamicFvMesh")
    if not isinstance(kind, Value) or _REFINEMENT_MESH_MARKER not in kind.text:
        return None
    return kind.text


def refuse_mesh_refinement() -> None:
    """Refuse a case whose ``dynamicFvMesh`` refines cells, before anything is built.

    Mesh *motion* is supported, adaptive refinement is not: nothing downstream
    reacts to cells appearing and disappearing, so an AMR case would silently solve
    on the initial topology. Call it as the *first* statement of a solver's ``mesh``
    init step, ahead of every ``context[...]`` lookup, so the case is refused before
    the argList, ``Foam::Time`` and mesh are constructed (:func:`create_mesh` does).
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

    Mirrors ``createDynamicFvMesh.H``, but branches on the dictionary rather than
    letting ``dynamicFvMesh::New`` fall back to a ``staticFvMesh``, so the
    static-mesh path stays on exactly the code it has always run. Call it behind
    :func:`refuse_mesh_refinement` (:func:`create_mesh` does).
    """
    if not Path("constant/dynamicMeshDict").is_file():
        return pyf.fvMesh(runtime)
    return pyf.dynamicFvMesh.New(arglist, runtime)


def create_mesh() -> InitStep:
    """The ``mesh`` init step — see :func:`new_mesh` for what it selects."""

    def create(context: dict[str, Any]) -> Any:
        # First, so an AMR case is refused before the context resolves the argList
        # and Foam::Time it would be built on.
        refuse_mesh_refinement()
        return new_mesh(context["models.foam_arglist"], context["runtime"])

    return lazy("mesh", create=create, depends_on=["runtime", "models.foam_arglist"])


def create_time_mesh(argv: list[str]) -> list[InitStep]:
    return [create_arglist(argv), create_runtime(), create_mesh()]


def read_vol_field(field_type: Type[Any], name: str) -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return field_type.read_field(context["mesh"], name)

    return field(name, create=create, depends_on=["mesh"])
