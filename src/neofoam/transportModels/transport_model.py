# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from typing import Any, Literal

from pydantic import BaseModel
from pybFoam.turbulence import singlePhaseTransportModel

from neofoam.core.plugin_system import PluginSystem


@PluginSystem.register(discriminator_variable="config", discriminator="transport_type")
class TransportModel(BaseModel):
    """
    Base class for transport property models.

    Provides extensibility for different transport models (single-phase,
    two-phase, non-Newtonian, etc.) within the IncompressibleFluid solver.
    """

    model_config = {"arbitrary_types_allowed": True}

    @property
    def provides(self) -> list[str]:
        """Fields this transport model adds to context."""
        return ["laminarTransport"]

    @property
    def requires(self) -> list[str]:
        """Fields this transport model needs."""
        return ["U", "phi"]

    @classmethod
    def create(cls, *, config: dict[str, Any]) -> Any:
        """Factory classmethod to create transport model from config.

        Implemented explicitly for type safety. Calls plugin_model generated
        by @PluginSystem.register decorator.
        """
        return cls.plugin_model(config=config)  # type: ignore[attr-defined]

    @classmethod
    def from_type(cls, transport_type: str) -> Any:
        """Create transport model config from type string.

        Args:
            transport_type: Type of transport model (e.g., 'singlePhase')

        Returns:
            Transport model config instance directly (not wrapped)

        Example:
            config = TransportModel.from_type('singlePhase')
            instance = config.create_instance(U, phi)
        """
        wrapper = cls.create(config={"transport_type": transport_type})
        return wrapper.config  # type: ignore[attr-defined]

    def create_instance(self, U: Any, phi: Any) -> Any:
        """
        Create the transport model instance.

        Args:
            U: Velocity field
            phi: Face flux field

        Returns:
            Transport model object
        """
        ...

    def setup(self, builder: Any) -> Any:
        """
        Create instance and register with builder.

        Uses builder.get_field() to retrieve required fields,
        then calls create_instance() and adds result to builder.
        """
        U = builder.get_field("U")
        phi = builder.get_field("phi")
        instance = self.create_instance(U, phi)
        builder.add_field("laminarTransport", instance)
        return instance


@TransportModel.register
class SinglePhaseTransport(BaseModel):
    """Single-phase Newtonian transport properties (default)."""

    transport_type: Literal["singlePhase"] = "singlePhase"
    model_config = {"arbitrary_types_allowed": True}

    def create_instance(self, U: Any, phi: Any) -> Any:
        """Create single-phase transport model using OpenFOAM."""
        return singlePhaseTransportModel(U, phi)
