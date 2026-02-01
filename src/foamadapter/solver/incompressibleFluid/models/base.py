# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Base class for SimpleSolver optional models.

Provides infrastructure for automatic model detection and integration.
"""

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from foamadapter.framework.initialization.lazy_init import LazyInit
from foamadapter.framework.operations import OperationCollection


@dataclass
class SimpleSolverModel(ABC):
    """
    Base class for SimpleSolver optional models.

    Models can extend SimpleSolver with additional physics:
    - Buoyancy (Boussinesq approximation)
    - Radiation
    - Species transport
    - Custom source terms

    Each model provides:
    - build(): LazyInit objects for fields/models
    - operations(): Operations to insert into execution graph
    """

    name: str
    enabled: bool = True

    @classmethod
    def detect_models(cls) -> list["SimpleSolverModel"]:
        """
        Scan for available models and instantiate configured ones.

        This method checks case files for model configuration and
        creates instances of enabled models.

        Returns:
            List of enabled model instances
        """
        from foamadapter.models.incompressible_fluid_model import (
            IncompressibleFluidModel,
        )

        # Reuse existing detection infrastructure
        # IncompressibleFluidModel.detect_models() scans for:
        # - g (gravity) -> Boussinesq
        # - radiationProperties -> Radiation
        # - etc.
        detected = IncompressibleFluidModel.detect_models()

        # Convert to SimpleSolver-compatible models
        simple_models = []
        for model in detected:
            if model.name == "boussinesq":
                from .boussinesq import BoussinesqModel

                simple_models.append(BoussinesqModel(name="boussinesq", enabled=True))
            # Add more model mappings as they're implemented

        return simple_models

    @abstractmethod
    def build(self) -> list[LazyInit]:
        """
        Build stage: Create lazy initializers for model fields.

        Returns list of LazyInit objects that will be executed
        during initialization to create fields and models.

        Example:
            return [
                field("T", depends_on=["mesh"], create=self._create_T),
                model("buoyancy", depends_on=["fields.T"], create=lambda _: self),
            ]
        """
        pass

    def operations(self) -> OperationCollection:
        """
        Return model-specific operations to insert into solver graph.

        These operations will be automatically merged with the main
        solver loop by the DAG resolver, respecting dependencies.

        Returns:
            OperationCollection with model operations
        """
        return OperationCollection()

    def resolve(self, config: Any) -> None:
        """
        Resolve stage: Connect with other models (optional).

        Called during initialization to establish inter-model
        dependencies through ConfigContext.

        Args:
            config: ConfigContext for inter-model communication
        """
        pass

    def validate(self) -> list[tuple[str, str]]:
        """
        Validate model configuration.

        Returns:
            List of (field, message) tuples for any validation errors
        """
        return []
