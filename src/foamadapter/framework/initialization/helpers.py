# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Helper Functions for Lazy Initialization

Provides convenience functions for creating LazyInit objects with common patterns.
"""

from typing import Callable, Any, List
from .lazy_init import LazyInit


def field(
    name: str, create: Callable[[], Any], depends_on: List[str] = None
) -> LazyInit:
    """
    Helper for creating field lazy initializers.

    Automatically prefixes name with "fields." and sets category.

    Args:
        name: Field name (e.g., "U", "p", "nu")
        create: Function that creates the field
        depends_on: List of dependencies (default: [])

    Returns:
        LazyInit for the field

    Example:
        field("U", create=lambda ctx: create_vector_field(ctx["mesh"], U0), depends_on=["mesh"])
        # Creates: LazyInit(name="fields.U", depends_on=["mesh"], ...)
    """
    if depends_on is None:
        depends_on = []

    return LazyInit(
        name=f"fields.{name}",
        depends_on=depends_on,
        initializer=create,
        category="fields",
    )


def operator(
    name: str, create: Callable[[], Any], depends_on: List[str] = None
) -> LazyInit:
    """
    Helper for creating operator lazy initializers.

    Automatically prefixes name with "operators." and sets category.

    Args:
        name: Operator name (e.g., "momentum", "pressure_poisson")
        create: Function that creates the operator
        depends_on: List of dependencies (required)

    Returns:
        LazyInit for the operator

    Example:
        operator("momentum",
                 depends_on=["fields.U", "fields.p"],
                 create=lambda: create_momentum_equation(mesh))
    """
    if depends_on is None:
        depends_on = []

    return LazyInit(
        name=f"operators.{name}",
        depends_on=depends_on,
        initializer=create,
        category="operators",
    )


def lazy(
    name: str, create: Callable[[], Any], depends_on: List[str] = None
) -> LazyInit:
    """
    General-purpose helper for creating lazy initializers.

    Use this for objects that don't fit the field/operator categories
    (e.g., mesh, runtime, solver loops).

    Args:
        name: Object name (e.g., "mesh", "runtime", "piso_loop")
        create: Function that creates the object
        depends_on: List of dependencies (default: [])

    Returns:
        LazyInit for the object

    Example:
        lazy("mesh", create=lambda: mesh)
        lazy("piso_loop",
             depends_on=["operators.momentum", "operators.pressure_poisson"],
             create=lambda: PISOLoop())
    """
    if depends_on is None:
        depends_on = []

    return LazyInit(name=name, depends_on=depends_on, initializer=create, category=None)


def model(
    name: str, create: Callable[[], Any], depends_on: List[str] = None
) -> LazyInit:
    """
    Helper for creating model instance lazy initializers.

    Automatically prefixes name with "models." and sets category.
    Use for components like transport models, turbulence models, etc.

    Args:
        name: Model name (e.g., "transport", "turbulence", "algorithm")
        create: Function that creates the model
        depends_on: List of dependencies

    Returns:
        LazyInit for the model

    Example:
        model("transport",
              depends_on=["fields.U", "fields.phi"],
              create=lambda: singlePhaseTransportModel(U, phi))
        model("turbulence",
              depends_on=["fields.U", "fields.phi", "models.transport"],
              create=lambda: incompressibleTurbulenceModel.New(U, phi, transport))
    """
    if depends_on is None:
        depends_on = []

    return LazyInit(
        name=f"models.{name}",
        depends_on=depends_on,
        initializer=create,
        category="models",
    )
