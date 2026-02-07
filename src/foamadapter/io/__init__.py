# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""IO utilities for configuration management."""

from foamadapter.io.strategies import (
    # Protocols
    ReadingStrategy,
    WritingStrategy,
    # Concrete strategies
    YAMLStrategy,
    JSONStrategy,
    # Registry
    IOStrategyRegistry,
    # Helper functions
    YAML,
    JSON,
    Custom,
    # Decorator
    IOStrategy,
    # Base class
    BaseConfig,
)

__all__ = [
    # Protocols
    "ReadingStrategy",
    "WritingStrategy",
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
]
