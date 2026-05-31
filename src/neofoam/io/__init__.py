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
    OF,
    IOStrategy,
)
from neofoam.io.base import BaseConfig
from neofoam.io.input_validation import validate_models
from neofoam.io.scaffold import collect_config_classes, save_configs

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
    "OF",
    # Decorator
    "IOStrategy",
    # Base class
    "BaseConfig",
    # Validation
    "ValidationErrors",
    "validate_models",
    # Scaffolding
    "save_configs",
    "collect_config_classes",
]
