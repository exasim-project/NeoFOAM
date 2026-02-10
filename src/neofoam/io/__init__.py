# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""IO utilities for configuration management."""

from neofoam.io.protocols import (
    ReadingStrategy,
    WritingStrategy,
)
from neofoam.io.strategies import (
    SubdictMixin,
    YAMLStrategy,
    JSONStrategy,
)
from neofoam.io.registry import IOStrategyRegistry
from neofoam.io.decorator import (
    YAML,
    JSON,
    Custom,
    IOStrategy,
)
from neofoam.io.base import BaseConfig
from neofoam.io.validation_types import ValidationErrors
from neofoam.io.input_validation import validate_models

__all__ = [
    # Protocols
    "ReadingStrategy",
    "WritingStrategy",
    # Subdict mixin
    "SubdictMixin",
    # Concrete strategies
    "YAMLStrategy",
    "JSONStrategy",
    # Registry
    "IOStrategyRegistry",
    # Helper functions
    "YAML",
    "JSON",
    "Custom",
    # Decorator
    "IOStrategy",
    # Base class
    "BaseConfig",
    # Validation
    "ValidationErrors",
    "validate_models",
]
