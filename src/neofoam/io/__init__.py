# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""IO utilities for configuration management."""

from neofoam.io.validation_types import (
    IOMetadata,
    ValidationErrors,
)
from neofoam.io.strategies import (
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
from neofoam.io.dictfile import DictFile
from neofoam.io.input_validation import validate_models
from neofoam.io.scaffold import collect_config_classes, save_configs
from neofoam.io.write_configs import write_configs
from neofoam.io.pydantic_schema import default_values, rjsf_uischema, slice_schema
from neofoam.io.schema import (
    ConfigInfo,
    ConfigSchema,
    ModelSummary,
    ToolInfo,
    config_schema,
    list_configs,
    model_catalog,
    tool_catalog,
)
from neofoam.io.dictread import (
    Unreadable,
    Value,
    read_entry,
    read_keys,
    read_section,
    read_toplevel,
)

__all__ = [
    # IO metadata
    "IOMetadata",
    # Format-marker strategies (dispatch is by file suffix in DictFile)
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
    # Batch config write
    "write_configs",
    # Format-agnostic dictionary-file handle (OpenFOAM-style interface)
    "DictFile",
    # JSON-schema form helpers
    "default_values",
    "rjsf_uischema",
    "slice_schema",
    # Config-introspection surface
    "ConfigInfo",
    "ConfigSchema",
    "ModelSummary",
    "ToolInfo",
    "config_schema",
    "list_configs",
    "model_catalog",
    "tool_catalog",
    # Typed OpenFOAM-dict leaf reader
    "Value",
    "Unreadable",
    "read_section",
    "read_entry",
    "read_keys",
    "read_toplevel",
]
