# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
IO strategies and decorator pattern for configuration files.

Provides:
- Reading/Writing strategy protocols
- YAML and JSON strategy implementations
- IOStrategyRegistry for managing strategies
- Helper functions (YAML, JSON, Custom)
- @IOStrategy decorator for declarative configuration
- BaseConfig with automatic strategy registration
"""

import yaml
import json
from typing import Any, Type, TypeVar, Protocol
from pathlib import Path
from pydantic import BaseModel
from foamadapter.io.input_validation import ModelInputDefinition

T = TypeVar("T", bound="BaseConfig")


class ReadingStrategy(Protocol):
    """Protocol for configuration reading strategies."""
    
    def read(self, path: Path, encoding: str = "utf-8") -> dict[str, Any]:
        """Read and parse configuration file."""
        ...


class WritingStrategy(Protocol):
    """Protocol for configuration writing strategies."""
    
    def write(self, data: dict[str, Any], path: Path, encoding: str = "utf-8") -> None:
        """Write configuration to file."""
        ...


class YAMLStrategy:
    """YAML reading and writing strategy.
    
    Supports three modes:
    1. Full file (subdict_path=None): Read/write entire YAML file
    2. Flat subdict (subdict_path="PIMPLE"): Read/write top-level key
    3. Nested subdict (subdict_path="solvers.p"): Read/write nested path with dot notation
    
    Examples:
        YAMLStrategy()  # Full file
        YAMLStrategy("PIMPLE")  # Flat subdict
        YAMLStrategy("solvers.p")  # Nested subdict
    """
    
    def __init__(self, subdict_path: str | None = None):
        """Initialize YAML strategy.
        
        Args:
            subdict_path: Optional path to subdict (e.g., "PIMPLE" or "solvers.p")
                         If contains dots, treated as nested path
        """
        self.subdict_path = subdict_path
        self.path_parts = subdict_path.split(".") if subdict_path else None
    
    def _get_nested(self, data: dict, path_parts: list[str]) -> dict:
        """Navigate to nested dict, returning empty dict if path doesn't exist."""
        current = data
        for part in path_parts:
            if not isinstance(current, dict) or part not in current:
                return {}
            current = current[part]
        return current if isinstance(current, dict) else {}
    
    def _set_nested(self, data: dict, path_parts: list[str], value: dict) -> None:
        """Set value at nested path, creating intermediate dicts as needed."""
        current = data
        for part in path_parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[path_parts[-1]] = value
    
    def read(self, path: Path, encoding: str = "utf-8") -> dict[str, Any]:
        """Read YAML configuration file."""
        if not path.exists():
            return {}
        with open(path, "r", encoding=encoding) as f:
            full_data = yaml.safe_load(f) or {}
        
        if not self.subdict_path:
            return full_data
        
        # Return subdict (flat or nested)
        if len(self.path_parts) == 1:
            return full_data.get(self.path_parts[0], {})
        else:
            return self._get_nested(full_data, self.path_parts)
    
    def write(self, data: dict[str, Any], path: Path, encoding: str = "utf-8") -> None:
        """Write configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.subdict_path:
            # Write full file
            with open(path, "w", encoding=encoding) as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            # Update subdict, preserving other sections
            existing_data = {}
            if path.exists():
                with open(path, "r", encoding=encoding) as f:
                    existing_data = yaml.safe_load(f) or {}
            
            if len(self.path_parts) == 1:
                existing_data[self.path_parts[0]] = data
            else:
                self._set_nested(existing_data, self.path_parts, data)
            
            with open(path, "w", encoding=encoding) as f:
                yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False)


class JSONStrategy:
    """JSON reading and writing strategy.
    
    Supports three modes:
    1. Full file (subdict_path=None): Read/write entire JSON file
    2. Flat subdict (subdict_path="database"): Read/write top-level key
    3. Nested subdict (subdict_path="services.database"): Read/write nested path with dot notation
    
    Examples:
        JSONStrategy()  # Full file
        JSONStrategy("database")  # Flat subdict
        JSONStrategy("services.database")  # Nested subdict
    """
    
    def __init__(self, subdict_path: str | None = None):
        """Initialize JSON strategy.
        
        Args:
            subdict_path: Optional path to subdict (e.g., "database" or "services.database")
                         If contains dots, treated as nested path
        """
        self.subdict_path = subdict_path
        self.path_parts = subdict_path.split(".") if subdict_path else None
    
    def _get_nested(self, data: dict, path_parts: list[str]) -> dict:
        """Navigate to nested dict, returning empty dict if path doesn't exist."""
        current = data
        for part in path_parts:
            if not isinstance(current, dict) or part not in current:
                return {}
            current = current[part]
        return current if isinstance(current, dict) else {}
    
    def _set_nested(self, data: dict, path_parts: list[str], value: dict) -> None:
        """Set value at nested path, creating intermediate dicts as needed."""
        current = data
        for part in path_parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[path_parts[-1]] = value
    
    def read(self, path: Path, encoding: str = "utf-8") -> dict[str, Any]:
        """Read JSON configuration file."""
        if not path.exists():
            return {}
        with open(path, "r", encoding=encoding) as f:
            full_data = json.load(f)
        
        if not self.subdict_path:
            return full_data
        
        # Return subdict (flat or nested)
        if len(self.path_parts) == 1:
            return full_data.get(self.path_parts[0], {})
        else:
            return self._get_nested(full_data, self.path_parts)
    
    def write(self, data: dict[str, Any], path: Path, encoding: str = "utf-8") -> None:
        """Write configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.subdict_path:
            # Write full file
            with open(path, "w", encoding=encoding) as f:
                json.dump(data, f, indent=2, sort_keys=False)
        else:
            # Update subdict, preserving other sections
            existing_data = {}
            if path.exists():
                with open(path, "r", encoding=encoding) as f:
                    existing_data = json.load(f)
            
            if len(self.path_parts) == 1:
                existing_data[self.path_parts[0]] = data
            else:
                self._set_nested(existing_data, self.path_parts, data)
            
            with open(path, "w", encoding=encoding) as f:
                json.dump(existing_data, f, indent=2)


class IOStrategyRegistry:
    """Registry for managing IO strategies for config classes."""
    
    def __init__(
        self,
        baseModel: Type[BaseModel],
        reading_strategy: ReadingStrategy | None = None,
        writing_strategy: WritingStrategy | None = None
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
        if not hasattr(baseModel, '__strategies__'):
            baseModel.__strategies__ = {}
        if not hasattr(baseModel, '__registries__'):
            baseModel.__registries__ = {}
            
        baseModel.__strategies__[baseModel.__name__] = (
            self.reading_strategy,
            self.writing_strategy
        )
        baseModel.__registries__[baseModel.__name__] = self
    
    def set_reading_strategy(self, strategy: ReadingStrategy) -> None:
        """Update reading strategy."""
        self.reading_strategy = strategy
        self.baseModel.__strategies__[self.baseModel.__name__] = (
            strategy,
            self.writing_strategy
        )
    
    def set_writing_strategy(self, strategy: WritingStrategy) -> None:
        """Update writing strategy."""
        self.writing_strategy = strategy
        self.baseModel.__strategies__[self.baseModel.__name__] = (
            self.reading_strategy,
            strategy
        )


# ============================================================================
# Decorator Pattern with Helpers - Variant 10
# ============================================================================

def YAML(file: str, required: bool = True, description: str = "", subdict: str | None = None) -> dict:
    """Helper to create YAML strategy configuration.
    
    Args:
        file: Relative path to YAML file
        required: Whether the file is required
        description: Description of this configuration
        subdict: Optional subdict path for partial file updates
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
        'input_file': file,
        'reading_strategy': strategy,
        'writing_strategy': strategy,
        'required': required,
        'description': description
    }


def JSON(file: str, required: bool = True, description: str = "", subdict: str | None = None) -> dict:
    """Helper to create JSON strategy configuration.
    
    Args:
        file: Relative path to JSON file
        required: Whether the file is required
        description: Description of this configuration
        subdict: Optional subdict path for partial file updates
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
        'input_file': file,
        'reading_strategy': strategy,
        'writing_strategy': strategy,
        'required': required,
        'description': description
    }


def Custom(
    file: str,
    reading_strategy: ReadingStrategy,
    writing_strategy: WritingStrategy | None = None,
    required: bool = True,
    description: str = ""
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
        'input_file': file,
        'reading_strategy': reading_strategy,
        'writing_strategy': writing_strategy or reading_strategy,
        'required': required,
        'description': description
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
            cls,
            config.get('reading_strategy'),
            config.get('writing_strategy')
        )
        
        # Create ModelInputDefinition from decorator config
        from dataclasses import replace
        input_definition = ModelInputDefinition(
            baseModel=cls,
            relative_path=config.get('input_file', f"{cls.__name__.lower()}.yaml"),
            required=config.get('required', True),
            description=config.get('description', '')
        )
        cls.register_input_definition(input_definition)
        
        return cls
    return decorator


class BaseConfig(BaseModel):
    """Base class for configurations with reading/writing strategies."""
    
    # Store strategies in a dict to avoid Pydantic field detection
    # Key is class name, value is (reading_strategy, writing_strategy) tuple
    __strategies__: dict[str, tuple] = {}
    
    # Store ModelInputDefinition for each config class
    # Key is class name, value is ModelInputDefinition
    __input_definitions__: dict[str, ModelInputDefinition] = {}
    
    # Store IOStrategyRegistry instances for dynamic updates
    __registries__: dict[str, "IOStrategyRegistry"] = {}
    
    def __init_subclass__(cls, **kwargs):
        """Automatically register strategies and input definition when subclass is defined."""
        super().__init_subclass__(**kwargs)
        
        # Note: Decorator pattern (@IOStrategy) runs AFTER __init_subclass__, 
        # so decorator handles its own registration.
    
    @classmethod
    def register_input_definition(cls, input_def: ModelInputDefinition) -> None:
        """Register ModelInputDefinition for this config class."""
        cls.__input_definitions__[cls.__name__] = input_def
    
    @classmethod
    def get_input_definition(cls) -> ModelInputDefinition | None:
        """Get ModelInputDefinition for this config class."""
        return cls.__input_definitions__.get(cls.__name__)
    
    @classmethod
    def get_default_path(cls, case_dir: Path | str = ".") -> Path:
        """Get default file path from registered ModelInputDefinition."""
        input_def = cls.get_input_definition()
        if not input_def:
            raise ValueError(f"No ModelInputDefinition registered for {cls.__name__}")
        return Path(case_dir) / input_def.relative_path
    
    @classmethod
    def get_reading_strategy(cls) -> ReadingStrategy:
        """Get the reading strategy for this config class."""
        return cls.__strategies__[cls.__name__][0]
    
    @classmethod
    def get_writing_strategy(cls) -> WritingStrategy:
        """Get the writing strategy for this config class."""
        return cls.__strategies__[cls.__name__][1]
    
    @classmethod
    def set_reading_strategy(cls, strategy: ReadingStrategy) -> None:
        """Set custom reading strategy for this config class."""
        cls.__registries__[cls.__name__].set_reading_strategy(strategy)
    
    @classmethod
    def set_writing_strategy(cls, strategy: WritingStrategy) -> None:
        """Set custom writing strategy for this config class."""
        cls.__registries__[cls.__name__].set_writing_strategy(strategy)

    @classmethod
    def load(cls: Type[T], case_dir: Path | str = ".", encoding: str = "utf-8") -> T:
        """Load configuration from file using registered strategy.
        
        Args:
            case_dir: Directory containing the configuration file or explicit file path
            encoding: File encoding (default: utf-8)
            
        Returns:
            Configuration instance loaded from file
        """
        # If case_dir is a file, use it directly
        case_dir_path = Path(case_dir)
        if case_dir_path.is_file() or (not case_dir_path.exists() and case_dir_path.suffix):
            path = case_dir_path
        else:
            path = cls.get_default_path(case_dir)
        
        reading_strategy = cls.get_reading_strategy()
        data = reading_strategy.read(path, encoding=encoding)
        return cls(**data)
    
    def save(self, case_dir: Path | str = ".", encoding: str = "utf-8") -> None:
        """Save configuration to file using registered strategy.
        
        Args:
            case_dir: Directory to save the configuration file or explicit file path
            encoding: File encoding (default: utf-8)
        """
        # If case_dir is a file, use it directly
        case_dir_path = Path(case_dir)
        if case_dir_path.suffix:  # Has file extension, treat as explicit path
            path = case_dir_path
        else:
            path = self.__class__.get_default_path(case_dir)
        
        writing_strategy = self.__class__.get_writing_strategy()
        data = self.model_dump(mode="python", exclude_none=False)
        writing_strategy.write(data, path, encoding=encoding)
