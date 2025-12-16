# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Buoyancy Model Plugin

Adds buoyancy effects to incompressible solvers through thermal expansion.
This is an example plugin model demonstrating the full lifecycle and
operation contribution pattern.

Example Usage:
    from foamadapter.models.buoyancy import BuoyancyModel

    solver = (
        IncompressibleFluid(argv=["buoyantCavity"])
        .add_model(BuoyancyModel(beta=3e-3, TRef=300.0))
    )
    ctx = solver.initialize()
    solver.run(ctx)
"""

from typing import Any

import pybFoam as pyf
from pybFoam import volScalarField, dimensionedVector, dimensionedScalar
from pydantic import BaseModel

from foamadapter.framework.context import FieldUpdates
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model
from foamadapter.framework.operations import Operation, OperationCollection
from foamadapter.models import register_model


@register_model("buoyancy")
@Model
class BuoyancyModel(BaseModel):
    """
    Adds buoyancy effects via thermal expansion: rho = rho0 * (1 - beta * (T - TRef))

    This model:
    - Reads temperature field during SETUP
    - Creates rhok (density correction) field
    - Contributes buoyancy source term to momentum equation

    Configuration:
        beta: Thermal expansion coefficient [1/K]
        TRef: Reference temperature [K]
        g: Gravity vector [m/s²]
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "buoyancy"
    beta: float = 3e-3  # Thermal expansion coefficient [1/K]
    TRef: float = 300.0  # Reference temperature [K]
    g: tuple[float, float, float] = (0, -9.81, 0)  # Gravity vector

    # Lifecycle state
    configured: bool = False

    @Model.load
    def load_buoyancy_properties(self) -> None:
        """
        LOAD: Load buoyancy properties from constant/buoyancyProperties.

        Falls back to defaults if file doesn't exist.
        """
        try:
            props = pyf.dictionary.read("constant/buoyancyProperties")
            self.beta = props.get[float]("beta")
            self.TRef = props.get[float]("TRef")

            # Read gravity vector if present
            if props.found("g"):
                g_vec = props.get("g")
                self.g = (g_vec.x(), g_vec.y(), g_vec.z())
        except (FileNotFoundError, KeyError):
            # Use constructor defaults
            pass

    @Model.resolve_dependencies
    def configure_buoyancy(self, registry: Any) -> None:
        """
        RESOLVE_DEPENDENCIES: Validate dependencies.

        Buoyancy model doesn't strictly require other models, but could
        check for compatibility with solver type.
        """
        self.configured = True

    @Model.build
    def setup_buoyancy_fields(self, builder: Any, mesh: Any = None) -> None:
        """  
        BUILD: Create temperature field and density correction.

        Reads T from disk and computes rhok = 1 - beta*(T - TRef).
        """
        # Read temperature field
        T = volScalarField.read_field(mesh, "T")
        builder.add_field("T", T)

        # Create density correction: rhok = 1 - beta*(T - TRef)
        # This is used in the buoyancy source term
        beta_dim = dimensionedScalar(
            "beta", pyf.dimless / pyf.dimTemperature, self.beta
        )
        TRef_dim = dimensionedScalar("TRef", pyf.dimTemperature, self.TRef)

        rhok = 1.0 - beta_dim * (T - TRef_dim)
        rhok.rename("rhok")

        builder.add_field("rhok", rhok)
        builder.add_model("buoyancy", self)

    @Model.operation(operation_number="1.5", depends_on=["momentum"])
    def buoyancy_source(self, U: Any, rhok: Any, UEqn: Any, ctx: Any) -> FieldUpdates:
        """
        Add buoyancy source term to momentum equation.

        Modifies UEqn with the term: rhok * g
        This runs after momentum assembly but before pressure correction.
        """
        # Create gravity vector as dimensioned quantity
        g = dimensionedVector(
            "g", pyf.dimAcceleration, (self.g[0], self.g[1], self.g[2])
        )

        # Add buoyancy source: UEqn -= rhok * g
        # Note: The minus sign is because rhok = 1 - beta*(T-TRef),
        # so hot fluid (T > TRef) has rhok < 1, creating upward force
        UEqn -= rhok * g

        return FieldUpdates({"UEqn": UEqn})

    def operations(self) -> OperationCollection:
        """
        Return operations this model contributes.

        Automatically discovers all @Model.operation decorated methods.
        """
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops
