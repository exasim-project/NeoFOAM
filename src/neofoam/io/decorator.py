# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""IOStrategy decorator and helper functions (YAML, JSON, Custom)."""

from typing import Optional

from neofoam.io.protocols import ReadingStrategy, WritingStrategy
from neofoam.io.strategies import YAMLStrategy, JSONStrategy
from neofoam.io.registry import IOStrategyRegistry
from neofoam.io.validation_types import ModelInputDefinition


def YAML(
    file: str,
    required: bool = True,
    description: str = "",
    subdict: Optional[str] = None,
) -> dict:
    """Helper to create YAML strategy configuration.

    Args:
        file: Relative path to YAML file
        required: Whether the file is required
        description: Description of this configuration
        subdict: Optional subdict path for partial file updates.
                 Examples: "PIMPLE" (flat) or "solvers.p" (nested with dots)

    Returns:
        Dictionary with strategy configuration

    Example:
        @IOStrategy(YAML("solver_config.yaml"))
        class SolverConfig(BaseConfig):
            param1: float

        @IOStrategy(YAML("fvSolution.yaml", subdict="PIMPLE"))
        class PIMPLEConfig(BaseConfig):
            nOuterCorrectors: int

        @IOStrategy(YAML("fvSolution.yaml", subdict="solvers.p"))
        class PSolverConfig(BaseConfig):
            solver: str
    """
    strategy = YAMLStrategy(subdict)

    return {
        "input_file": file,
        "reading_strategy": strategy,
        "writing_strategy": strategy,
        "required": required,
        "description": description,
    }


def JSON(
    file: str,
    required: bool = True,
    description: str = "",
    subdict: Optional[str] = None,
) -> dict:
    """Helper to create JSON strategy configuration.

    Args:
        file: Relative path to JSON file
        required: Whether the file is required
        description: Description of this configuration
        subdict: Optional subdict path for partial file updates.
                 Examples: "database" (flat) or "services.database" (nested with dots)

    Returns:
        Dictionary with strategy configuration

    Example:
        @IOStrategy(JSON("config.json"))
        class MyConfig(BaseConfig):
            setting: str

        @IOStrategy(JSON("config.json", subdict="services.database"))
        class DatabaseConfig(BaseConfig):
            host: str
    """
    strategy = JSONStrategy(subdict)

    return {
        "input_file": file,
        "reading_strategy": strategy,
        "writing_strategy": strategy,
        "required": required,
        "description": description,
    }


def Custom(
    file: str,
    reading_strategy: ReadingStrategy,
    writing_strategy: Optional[WritingStrategy] = None,
    required: bool = True,
    description: str = "",
) -> dict:
    """Helper to create custom strategy configuration.

    Args:
        file: Relative path to file
        reading_strategy: Custom reading strategy
        writing_strategy: Custom writing strategy (defaults to reading_strategy)
        required: Whether the file is required
        description: Description of this configuration

    Returns:
        Dictionary with strategy configuration

    Example:
        @IOStrategy(Custom("data.txt", MyCustomStrategy()))
        class CustomConfig(BaseConfig):
            data: str
    """
    return {
        "input_file": file,
        "reading_strategy": reading_strategy,
        "writing_strategy": writing_strategy or reading_strategy,
        "required": required,
        "description": description,
    }


def IOStrategy(config: dict):
    """Decorator to set IO strategy on a configuration class.

    Stores strategy configuration in the class's __io_strategy__ attribute,
    which is then read by BaseConfig.__init_subclass__ for automatic registration.

    Args:
        config: Strategy configuration dictionary (usually from YAML/JSON/Custom helper)

    Returns:
        Decorator function

    Example:
        @IOStrategy(YAML("solver_config.yaml"))
        class SolverConfig(BaseConfig):
            param1: float
            param2: float
    """

    def decorator(cls):
        # Store strategy configuration in private attribute
        cls.__io_strategy__ = config

        # Manually trigger registration since decorator runs after __init_subclass__
        IOStrategyRegistry(
            cls, config.get("reading_strategy"), config.get("writing_strategy")
        )

        # Create ModelInputDefinition from decorator config
        input_definition = ModelInputDefinition(
            baseModel=cls,
            relative_path=config.get("input_file", f"{cls.__name__.lower()}.yaml"),
            required=config.get("required", True),
            description=config.get("description", ""),
        )
        cls.register_input_definition(input_definition)

        return cls

    return decorator
