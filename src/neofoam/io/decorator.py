# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""IOStrategy decorator and helper functions (YAML, JSON)."""

from typing import Any, Callable, Optional, TypeVar

from neofoam.io.validation_types import IOMetadata
from neofoam.io.strategies import YAMLStrategy, JSONStrategy, OpenFOAMStrategy


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


def OF(
    file: str,
    subdict: Optional[str] = None,
) -> dict[str, Any]:
    """Helper to create OpenFOAM dictionary strategy configuration.

    Args:
        file: Relative path to OpenFOAM dictionary file
        subdict: Optional subdict path for partial file updates.
                 Examples: "PISO" (flat) or "solvers.p" (nested with dots)

    Returns:
        Dictionary with strategy configuration
    """
    strategy = OpenFOAMStrategy(subdict)

    return {
        "input_file": file,
        "reading_strategy": strategy,
        "writing_strategy": strategy,
    }


T = TypeVar("T")


def IOStrategy(config: dict[str, Any]) -> Callable[[type[T]], type[T]]:
    """Decorator to set IO strategy on a configuration class.

    Sets ``io_config`` on the decorated class to an :class:`IOMetadata` instance.

    Args:
        config: Strategy configuration dictionary (usually from YAML/JSON helper)

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
