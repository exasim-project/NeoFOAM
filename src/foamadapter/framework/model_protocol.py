# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
SolverModel Protocol

Defines the interface for modular, pluggable models that extend solver capabilities.
Models implementing this protocol can be composed into solvers to add features like
turbulence, buoyancy, radiation, species transport, etc.

Example:
    @Model
    class BuoyancyModel(BaseModel):
        name: str = "buoyancy"
        beta: float = 3e-3

        @Model.read_files
        def load_properties(self):
            ...

        @Model.configure
        def connect_models(self, registry):
            ...

        @Model.setup
        def setup_fields(self, mesh, builder):
            builder.add_field("T", temperature_field)
            builder.add_model("buoyancy", self)

        @Model.operation(operation_number=1, depends_on=["momentum"])
        def buoyancy_correction(self, U, T) -> FieldUpdates:
            ...

        def operations(self) -> OperationCollection:
            ...

    # Usage
    solver = (
        IncompressibleFluid(argv=["cavity"])
        .add_model(BuoyancyModel(beta=3e-3))
    )
    ctx = solver.initialize()
"""

from typing import Protocol, runtime_checkable, Any
from foamadapter.framework.operations import OperationCollection


@runtime_checkable
class SolverModel(Protocol):
    """
    Protocol for models that extend solver capabilities.

    Models implementing this protocol participate in the 3-stage initialization
    lifecycle and contribute operations to the solver's execution graph.

    Lifecycle Stages:
        1. READ_FILES: Load configuration from files (optional)
        2. CONFIGURE: Validate and connect to other models via ModelRegistry (optional)
        3. SETUP: Create runtime objects and add to ContextBuilder (required)

    Attributes:
        name: Unique identifier for this model
    """

    @property
    def name(self) -> str:
        """
        Unique name for this model.

        Used for model registration, lookups, and debugging. Should be
        a valid Python identifier (lowercase, underscores).

        Examples:
            "buoyancy", "turbulence", "radiation", "species_transport"
        """
        ...

    def read_files(self) -> None:
        """
        READ_FILES stage: Load configuration from files.

        Optional lifecycle method. Mark implementation with @Model.read_files decorator.
        Loads properties, parameters, or data from files before initialization.

        Example:
            @Model.read_files
            def load_properties(self):
                props = dictionary.read("constant/buoyancyProperties")
                self.beta = props.get[float]("beta")
        """
        ...

    def configure(self, registry: Any) -> None:
        """
        CONFIGURE stage: Validate and connect to other models.

        Optional lifecycle method. Mark implementation with @Model.configure decorator.
        Use ModelRegistry to query and adapt other models' behavior.

        Args:
            registry: ModelRegistry for inter-model communication

        Example:
            @Model.configure
            def configure(self, registry):
                pressure = registry.get("pressure_algorithm")
                if pressure:
                    pressure.use_buoyancy = True  # Enable buoyancy variant
        """
        ...

    def setup(self, mesh: Any, builder: Any) -> None:
        """
        SETUP stage: Create runtime objects and add to context.

        Required lifecycle method. Mark implementation with @Model.setup decorator.
        Create fields, read initial conditions, and register contributions.

        Args:
            mesh: The computational mesh
            builder: ContextBuilder to register fields and models

        Example:
            @Model.setup
            def setup_fields(self, mesh, builder):
                T = volScalarField.read_field(mesh, "T")
                builder.add_field("T", T)
                builder.add_model("buoyancy", self)
        """
        ...

    def operations(self) -> OperationCollection:
        """
        Return operations this model contributes.

        Operations define computational steps (momentum correction, field updates,
        boundary conditions, etc.) that execute during the main loop.

        Returns:
            OperationCollection containing this model's operations

        Example:
            def operations(self) -> OperationCollection:
                from foamadapter.framework.decorator import decorated_member_functions
                funcs = decorated_member_functions(self)
                ops = OperationCollection()
                for func in funcs:
                    ops.add(Operation.create_SeqOp(func))
                return ops
        """
        ...
