# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM-specific initialization helpers."""

from typing import Any, Type

import pybFoam as pyf

from neofoam.framework.initialization import InitStep, field, lazy, model


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


def create_mesh() -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return pyf.fvMesh(context["runtime"])

    return lazy("mesh", create=create, depends_on=["runtime"])


def create_time_mesh(argv: list[str]) -> list[InitStep]:
    return [create_arglist(argv), create_runtime(), create_mesh()]


def read_vol_field(field_type: Type[Any], name: str) -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return field_type.read_field(context["mesh"], name)

    return field(name, create=create, depends_on=["mesh"])
