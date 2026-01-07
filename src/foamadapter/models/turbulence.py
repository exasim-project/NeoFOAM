# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from foamadapter.core.plugin_system import PluginSystem
from pydantic import BaseModel
from typing import Any, Literal
from pybFoam.turbulence import incompressibleTurbulenceModel


@PluginSystem.register(discriminator_variable="config", discriminator="turbulence_type")
class TurbulenceModel(BaseModel):
    """
    Base class for turbulence models.

    Provides extensibility for different turbulence models (OpenFOAM RTS,
    laminar, custom implementations) within the IncompressibleFluid solver.
    """

    model_config = {"arbitrary_types_allowed": True}

    @property
    def provides(self) -> list[str]:
        """Fields this turbulence model adds to context."""
        return ["turbulence"]

    @property
    def requires(self) -> list[str]:
        """Fields this turbulence model needs."""
        return ["U", "phi", "laminarTransport"]

    @classmethod
    def create(cls, *, config: dict[str, Any]) -> Any:
        """Factory classmethod to create turbulence model from config.

        Implemented explicitly for type safety. Calls plugin_model generated
        by @PluginSystem.register decorator.
        """
        return cls.plugin_model(config=config)  # type: ignore[attr-defined]

    @classmethod
    def from_type(cls, turbulence_type: str) -> Any:
        """Create turbulence model config from type string.

        Args:
            turbulence_type: Type of turbulence model (e.g., 'openfoam_rts', 'laminar')

        Returns:
            Turbulence model config instance directly (not wrapped)

        Example:
            config = TurbulenceModel.from_type('openfoam_rts')
            instance = config.create_instance(U, phi, transport)
        """
        wrapper = cls.create(config={"turbulence_type": turbulence_type})
        return wrapper.config  # type: ignore[attr-defined]

    def create_instance(self, U: Any, phi: Any, transport: Any) -> Any:
        """
        Create the turbulence model instance.

        Args:
            U: Velocity field
            phi: Face flux field
            transport: Transport model

        Returns:
            Turbulence model object
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
        transport = builder.get_field("laminarTransport")
        turbulence = self.create_instance(U, phi, transport)
        builder.add_field("turbulence", turbulence)
        return turbulence


@TurbulenceModel.register
class OpenFOAMTurbulence(BaseModel):
    """Wrapper for OpenFOAM's turbulence models (default)."""

    turbulence_type: Literal["openfoam_rts"] = "openfoam_rts"
    model_config = {"arbitrary_types_allowed": True}

    def create_instance(self, U: Any, phi: Any, transport: Any) -> Any:
        """Create turbulence model using OpenFOAM's runtime selection."""
        return incompressibleTurbulenceModel.New(U, phi, transport)


@TurbulenceModel.register
class LaminarModel(BaseModel):
    """Explicit laminar (no turbulence) - useful for testing."""

    turbulence_type: Literal["laminar"] = "laminar"
    model_config = {"arbitrary_types_allowed": True}

    def create_instance(self, U: Any, phi: Any, transport: Any) -> Any:
        """
        Create a mock laminar turbulence model for testing.

        Note: This returns a minimal turbulence object. For production use,
        use OpenFOAM's laminar model via openfoam type.
        """
        # For now, return OpenFOAM's turbulence model which will read
        # the laminar model from constant/momentumTransport
        return incompressibleTurbulenceModel.New(U, phi, transport)
