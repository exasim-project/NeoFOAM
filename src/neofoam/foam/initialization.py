# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""OpenFOAM-specific initialization helpers."""

from typing import Any, Type

import pybFoam as pyf

from neofoam.framework.initialization import InitStep, field, init


def create_arg_list(argv: list[str]) -> InitStep:
    def create(_context: dict[str, Any]) -> Any:
        return pyf.argList(argv)

    return init("arg_list", create=create)


def create_runtime() -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return pyf.Time(context["arg_list"])

    return init("runtime", create=create, depends_on=["arg_list"])


def create_mesh() -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return pyf.fvMesh(context["runtime"])

    return init("mesh", create=create, depends_on=["runtime"])


def create_time_mesh(argv: list[str]) -> list[InitStep]:
    return [create_arg_list(argv), create_runtime(), create_mesh()]


def read_vol_field(field_type: Type[Any], name: str) -> InitStep:
    def create(context: dict[str, Any]) -> Any:
        return field_type.read_field(context["mesh"], name)

    return field(name, create=create, depends_on=["mesh"])
