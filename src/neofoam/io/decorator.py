# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""IOStrategy decorator and helper functions (YAML, JSON, Custom)."""

from typing import Any, Callable, Optional, TypeVar

from neofoam.io.validation_types import IOMetadata, ReadingStrategy, WritingStrategy
from neofoam.io.strategies import YAMLStrategy, JSONStrategy


def YAML(
    file: str,
    subdict: Optional[str] = None,
) -> dict[str, Any]:
    """Helper to create YAML strategy configuration.

    Args:
        file: Relative path to YAML file
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
    }


def JSON(
    file: str,
    subdict: Optional[str] = None,
) -> dict[str, Any]:
    """Helper to create JSON strategy configuration.

    Args:
        file: Relative path to JSON file
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
    }


def Custom(
    file: str,
    reading_strategy: ReadingStrategy,
    writing_strategy: Optional[WritingStrategy] = None,
) -> dict[str, Any]:
    """Helper to create custom strategy configuration.

    Args:
        file: Relative path to file
        reading_strategy: Custom reading strategy
        writing_strategy: Custom writing strategy (defaults to reading_strategy)

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
    }


T = TypeVar("T")


def IOStrategy(config: dict[str, Any]) -> Callable[[type[T]], type[T]]:
    """Decorator to set IO strategy on a configuration class.

    Sets ``io_config`` on the decorated class to an :class:`IOMetadata` instance.

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
    reader = config.get("reading_strategy", YAMLStrategy())

    def decorator(cls: type[T]) -> type[T]:
        cls.io_config = IOMetadata(  # type: ignore[attr-defined]
            file=config.get("input_file", f"{cls.__name__.lower()}.yaml"),
            reader=reader,
            writer=config.get("writing_strategy", reader),
        )
        return cls

    return decorator
