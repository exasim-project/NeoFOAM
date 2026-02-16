# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""IO utilities for configuration management."""

from neofoam.io.validation_types import (
    ReadingStrategy,
    WritingStrategy,
    IOMetadata,
    ValidationErrors,
)
from neofoam.io.strategies import (
    SubdictMixin,
    YAMLStrategy,
    JSONStrategy,
    OpenFOAMStrategy,
)
from neofoam.io.decorator import (
    YAML,
    JSON,
    OPENFOAM,
    IOStrategy,
)
from neofoam.io.base import BaseConfig
from neofoam.io.input_validation import validate_models

__all__ = [
    # Protocols
    "ReadingStrategy",
    "WritingStrategy",
    # IO metadata
    "IOMetadata",
    # Subdict mixin
    "SubdictMixin",
    # Concrete strategies
    "YAMLStrategy",
    "JSONStrategy",
    "OpenFOAMStrategy",
    # Helper functions
    "YAML",
    "JSON",
    "OPENFOAM",
    # Decorator
    "IOStrategy",
    # Base class
    "BaseConfig",
    # Validation
    "ValidationErrors",
    "validate_models",
]
