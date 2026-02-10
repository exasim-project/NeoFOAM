# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""IOStrategyRegistry for managing IO strategies for config classes."""

from typing import Optional, Type

from pydantic import BaseModel

from neofoam.io.protocols import ReadingStrategy, WritingStrategy
from neofoam.io.strategies import YAMLStrategy


class IOStrategyRegistry:
    """Registry for managing IO strategies for config classes."""

    def __init__(
        self,
        baseModel: Type[BaseModel],
        reading_strategy: Optional[ReadingStrategy] = None,
        writing_strategy: Optional[WritingStrategy] = None,
    ):
        """Initialize registry and set strategies for the given model.

        Args:
            baseModel: The configuration class to register strategies for
            reading_strategy: Strategy for reading config files (defaults to YAML)
            writing_strategy: Strategy for writing config files (defaults to YAML)
        """
        self.baseModel = baseModel
        self.reading_strategy = reading_strategy or YAMLStrategy()
        self.writing_strategy = writing_strategy or YAMLStrategy()

        # Register strategies with the model class
        if not hasattr(baseModel, "__strategies__"):
            baseModel.__strategies__ = {}
        if not hasattr(baseModel, "__registries__"):
            baseModel.__registries__ = {}

        baseModel.__strategies__[baseModel.__name__] = (
            self.reading_strategy,
            self.writing_strategy,
        )
        baseModel.__registries__[baseModel.__name__] = self

    def set_reading_strategy(self, strategy: ReadingStrategy) -> None:
        """Update reading strategy."""
        self.reading_strategy = strategy
        self.baseModel.__strategies__[self.baseModel.__name__] = (
            strategy,
            self.writing_strategy,
        )

    def set_writing_strategy(self, strategy: WritingStrategy) -> None:
        """Update writing strategy."""
        self.writing_strategy = strategy
        self.baseModel.__strategies__[self.baseModel.__name__] = (
            self.reading_strategy,
            strategy,
        )
