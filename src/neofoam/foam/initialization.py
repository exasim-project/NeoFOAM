# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM-specific initialization helpers."""

from typing import Any, Type

import pybFoam as pyf

from neofoam.framework.initialization import InitStep, field, lazy


def create_runtime(argv: list[str]) -> InitStep:
    def create(_context: dict[str, Any]) -> Any:
        arg_list = pyf.argList(argv)
        return pyf.Time(arg_list)

    return lazy("runtime", create=create)


def create_mesh() -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return pyf.fvMesh(context["runtime"])

    return lazy("mesh", create=create, depends_on=["runtime"])


def create_time_mesh(argv: list[str]) -> list[InitStep]:
    return [create_runtime(argv), create_mesh()]


def read_vol_field(field_type: Type[Any], name: str) -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return field_type.read_field(context["mesh"], name)

    return field(name, create=create, depends_on=["mesh"])
