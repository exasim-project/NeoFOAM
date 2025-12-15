# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""
Context Builder

Accumulates contributions from models and solver during the BUILD stage.
"""

from typing import Any


class ContextBuilder:
    """
    Collects contributions from models and solver during BUILD stage.

    The ContextBuilder accumulates fields, models, mesh, and runtime objects
    from various sources during initialization, then builds a complete Context.
    This enables a compositional approach where multiple models can contribute
    to the simulation state.

    Example:
        builder = ContextBuilder()

        # Models contribute fields
        builder.add_field("p", pressure_field)
        builder.add_field("U", velocity_field)

        # Set infrastructure
        builder.set_mesh(mesh)
        builder.set_runtime(runTime)

        # Build final context
        ctx = builder.build()
    """

    def __init__(self):
        """Initialize an empty builder."""
        self._fields: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._mesh: Any = None
        self._runTime: Any = None

    def add_field(self, name: str, field: Any) -> None:
        """
        Add a field to the context.

        Args:
            name: Field name (e.g., "p", "U", "phi")
            field: Field object (e.g., volScalarField, volVectorField)

        Raises:
            ValueError: If field name already exists
        """
        if name in self._fields:
            raise ValueError(
                f"Field '{name}' already exists in context. Cannot add duplicate field."
            )
        self._fields[name] = field

    def add_model(self, name: str, model: Any) -> None:
        """
        Add a model to the context.

        Args:
            name: Model name (e.g., "transport", "turbulence", "algorithm")
            model: Model instance

        Raises:
            ValueError: If model name already exists
        """
        if name in self._models:
            raise ValueError(
                f"Model '{name}' already exists in context. Cannot add duplicate model."
            )
        self._models[name] = model

    def set_mesh(self, mesh: Any) -> None:
        """
        Set the mesh object.

        Args:
            mesh: The finite volume mesh (fvMesh)

        Raises:
            ValueError: If mesh has already been set
        """
        if self._mesh is not None:
            raise ValueError(
                "Mesh already set in context builder. Mesh can only be set once."
            )
        self._mesh = mesh

    def set_runtime(self, runTime: Any) -> None:
        """
        Set the runtime object.

        Args:
            runTime: The simulation time manager

        Raises:
            ValueError: If runTime has already been set
        """
        if self._runTime is not None:
            raise ValueError(
                "Runtime already set in context builder. Runtime can only be set once."
            )
        self._runTime = runTime

    def build(self) -> "Context":
        """
        Build the final Context from accumulated contributions.

        Returns:
            A complete Context object ready for simulation

        Note:
            For production code, both mesh and runTime should be set.
            For unit tests without BUILD methods, they can be left as None.
        """
        from foamadapter.framework.context import Context

        return Context(
            fields=self._fields,
            models=self._models,
            mesh=self._mesh,
            runTime=self._runTime,
        )
