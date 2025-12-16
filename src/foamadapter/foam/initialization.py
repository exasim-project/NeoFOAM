# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""OpenFOAM-specific initialization helpers.

This module provides convenience functions for creating LazyInit objects
for common OpenFOAM runtime structures and fields.
"""

from typing import Any, Type

import pybFoam as pyf

from foamadapter.framework.initialization.helpers import field, lazy
from foamadapter.framework.initialization.lazy_init import LazyInit


def create_runtime(argv: list[str]) -> LazyInit:
    """
    Create a LazyInit for OpenFOAM runtime (Time object).

    Args:
        argv: Command-line arguments for argList

    Returns:
        LazyInit with no dependencies that creates Time object

    Example:
        >>> from foamadapter.foam.initialization import create_runtime, create_mesh
        >>> initializers = [
        ...     create_runtime(["solver", "-case", "cavity"]),
        ...     create_mesh(),
        ... ]
    """

    def create() -> Any:
        argList = pyf.argList(argv)
        return pyf.Time(argList)

    return lazy("runtime", create=create)


def create_mesh() -> LazyInit:
    """
    Create a LazyInit for OpenFOAM mesh (fvMesh object).

    Depends on runtime being initialized first.

    Returns:
        LazyInit that creates fvMesh from runtime

    Example:
        >>> from foamadapter.foam.initialization import create_runtime, create_mesh
        >>> initializers = [
        ...     create_runtime(["solver"]),
        ...     create_mesh(),
        ... ]
    """

    def create(context: dict[str, Any]) -> Any:
        runTime = context["runtime"]
        return pyf.fvMesh(runTime)

    return lazy("mesh", depends_on=["runtime"], create=create)


def create_time_mesh(argv: list[str]) -> list[LazyInit]:
    """
    Create LazyInit objects for both OpenFOAM runtime (Time) and mesh (fvMesh).

    This is a convenience function that groups runtime and mesh creation together.

    Args:
        argv: Command-line arguments for argList

    Returns:
        List containing LazyInit objects for runtime and mesh

    Example:
        >>> from foamadapter.foam.initialization import create_time_mesh
        >>> initializers = create_time_mesh(["solver", "-case", "cavity"])
    """
    return [
        create_runtime(argv),
        create_mesh(),
    ]


def read_vol_field(field_type: Type[Any], name: str) -> LazyInit:
    """
    Create LazyInit for reading a volumetric field from mesh.

    Args:
        field_type: OpenFOAM field type (volScalarField, volVectorField, etc.)
        name: Field name to read

    Returns:
        LazyInit that reads the field from disk

    Example:
        >>> from foamadapter.foam.initialization import read_vol_field
        >>> from pybFoam import volScalarField, volVectorField
        >>> initializers = [
        ...     read_vol_field(volScalarField, "p"),
        ...     read_vol_field(volVectorField, "U"),
        ... ]
    """

    def create(context: dict[str, Any]) -> Any:
        mesh = context["mesh"]
        return field_type.read_field(mesh, name)

    return field(name, create, depends_on=["mesh"])
