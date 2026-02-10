# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""BaseConfig — Pydantic model with automatic IO strategy registration."""

from typing import Any, Optional, Type, TypeVar, Union
from pathlib import Path

from pydantic import BaseModel, ValidationError

from neofoam.io.protocols import ReadingStrategy, WritingStrategy
from neofoam.io.validation_types import ModelInputDefinition, ValidationErrors

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base class for configurations with reading/writing strategies."""

    # Store strategies in a dict to avoid Pydantic field detection
    # Key is class name, value is (reading_strategy, writing_strategy) tuple
    __strategies__: dict[str, tuple] = {}

    # Store ModelInputDefinition for each config class
    # Key is class name, value is ModelInputDefinition
    __input_definitions__: dict[str, ModelInputDefinition] = {}

    # Store IOStrategyRegistry instances for dynamic updates
    __registries__: dict[str, Any] = {}

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
    def get_input_definition(cls) -> Optional[ModelInputDefinition]:
        """Get ModelInputDefinition for this config class."""
        return cls.__input_definitions__.get(cls.__name__)

    @classmethod
    def get_default_path(cls, case_dir: Union[Path, str] = ".") -> Path:
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
    def load(
        cls: Type[T],
        case_dir: Union[Path, str] = ".",
        encoding: str = "utf-8",
        validate: bool = True,
        file: Optional[str] = None,
    ) -> T:
        """Load configuration from file using registered strategy.

        Reads the file via the registered strategy, then optionally validates
        the data against the model's field types and constraints.

        When ``validate=True`` (default), data is loaded and validated via
        ``model_validate``. This ensures **all** validation errors are
        captured and reported in a single ``ValidationError``.

        When ``validate=False``, data is loaded via ``model_construct`` only,
        skipping validation entirely (useful for inspection or migration).

        Args:
            case_dir: Directory containing the configuration file or explicit file path
            encoding: File encoding (default: utf-8)
            validate: If True, validate data after loading and raise
                      ``pydantic.ValidationError`` on failure (default: True)
            file: Optional filename override. When provided, resolves relative
                  to ``case_dir`` instead of using the registered default path.

        Returns:
            Configuration instance loaded from file

        Raises:
            FileNotFoundError: If the configuration file does not exist
            KeyError: If a configured subdict path doesn't resolve
            pydantic.ValidationError: If validate=True and data has errors
        """
        case_dir_path = Path(case_dir)
        if case_dir_path.is_file() or (
            not case_dir_path.exists() and case_dir_path.suffix
        ):
            # case_dir is an explicit file path
            path = case_dir_path
        elif file is not None:
            # Override filename, resolve relative to case_dir
            path = case_dir_path / file
        else:
            path = cls.get_default_path(case_dir)

        reading_strategy = cls.get_reading_strategy()
        data = reading_strategy.read(path, encoding=encoding)

        if validate:
            return cls.model_validate(data)

        return cls.model_construct(**data)

    @property
    def configs(self) -> list["BaseConfig"]:
        """Convenience property to get all config models together."""
        return [self]

    @property
    def file_name(self) -> str:
        """Get the file name for this configuration from ModelInputDefinition.

        Returns:
            Relative path to the configuration file

        Example:
            >>> config = SolverConfig(...)
            >>> config.file_name
            'solver_config.yaml'
        """
        input_def = self.__class__.get_input_definition()
        return input_def.relative_path if input_def else self.__class__.__name__

    @property
    def subdict(self) -> Optional[str]:
        """Get the optional subdict path from the reading strategy.

        Returns:
            Subdict path if configured, None otherwise

        Example:
            >>> pimple_config = PIMPLEConfig(...)
            >>> pimple_config.subdict
            'PIMPLE'

            >>> config = SolverConfig(...)
            >>> config.subdict
            None
        """
        try:
            reading_strategy = self.__class__.get_reading_strategy()
            if hasattr(reading_strategy, "subdict_path"):
                return reading_strategy.subdict_path
        except (KeyError, AttributeError):
            pass
        return None

    @property
    def metadata(self) -> tuple[str, Optional[str]]:
        """Get both file name and subdict as a tuple.

        Returns:
            Tuple of (file_name, subdict)

        Example:
            >>> config = PIMPLEConfig(...)
            >>> config.metadata
            ('fvSolution.yaml', 'PIMPLE')
        """
        return self.file_name, self.subdict

    def save(
        self,
        case_dir: Union[Path, str] = ".",
        encoding: str = "utf-8",
        file: Optional[str] = None,
    ) -> None:
        """Save configuration to file using registered strategy.

        Args:
            case_dir: Directory to save the configuration file or explicit file path
            encoding: File encoding (default: utf-8)
            file: Optional filename override. When provided, resolves relative
                  to ``case_dir`` instead of using the registered default path.
        """
        case_dir_path = Path(case_dir)
        if case_dir_path.suffix:  # Has file extension, treat as explicit path
            path = case_dir_path
        elif file is not None:
            path = case_dir_path / file
        else:
            path = self.__class__.get_default_path(case_dir)

        writing_strategy = self.__class__.get_writing_strategy()
        data = self.model_dump(mode="python", exclude_none=False)
        writing_strategy.write(data, path, encoding=encoding)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @classmethod
    def _wrap_errors(
        cls,
        exc: ValidationError,
        file_override: Optional[str] = None,
    ) -> list[ValidationErrors]:
        """Convert a ``ValidationError`` into a list of ``ValidationErrors``.

        Shared helper used by :meth:`collect_errors` and :meth:`validate`.

        Args:
            exc: The Pydantic ``ValidationError`` to convert.
            file_override: If provided, use this as the ``file_name`` context
                instead of the registered default.

        Returns:
            List of ``ValidationErrors`` with file and subdict context.
        """
        input_def = cls.get_input_definition()
        file_name = file_override or (
            input_def.relative_path if input_def else cls.__name__
        )

        # Resolve subdict from reading strategy
        subdict: Optional[str] = None
        try:
            reading_strategy = cls.get_reading_strategy()
            if hasattr(reading_strategy, "subdict_path"):
                subdict = reading_strategy.subdict_path
        except (KeyError, AttributeError):
            pass

        return [
            ValidationErrors(
                field=err.get("loc", [None]),
                message=err.get("msg", "Validation error"),
                file_name=file_name,
                input_value=err.get("input", None),
                error_type=err.get("type", "UnknownError"),
                subdict=subdict,
            )
            for err in exc.errors(include_url=False)
        ]

    @classmethod
    def collect_errors(
        cls,
        case_dir: Union[Path, str] = ".",
        encoding: str = "utf-8",
        file: Optional[str] = None,
    ) -> list[ValidationErrors]:
        """Load and validate this config, returning errors instead of raising.

        Performs the same load as ``load(validate=True)`` but catches
        ``FileNotFoundError``, ``KeyError``, and ``ValidationError`` and
        converts them to a list of ``ValidationErrors`` with file/subdict
        context.

        Returns an empty list when validation succeeds.

        Args:
            case_dir: Directory containing the configuration file
            encoding: File encoding (default: utf-8)
            file: Optional filename override (see :meth:`load`)

        Returns:
            List of ``ValidationErrors`` (empty if valid)
        """
        input_def = cls.get_input_definition()
        file_name = file or (
            input_def.relative_path if input_def else cls.__name__
        )

        try:
            cls.load(case_dir, encoding=encoding, validate=True, file=file)
            return []
        except FileNotFoundError as exc:
            return [
                ValidationErrors(
                    field=None,
                    message=str(exc),
                    file_name=file_name,
                    error_type="FileNotFound",
                )
            ]
        except KeyError as exc:
            return [
                ValidationErrors(
                    field=None,
                    message=str(exc),
                    file_name=file_name,
                    error_type="KeyError",
                )
            ]
        except ValidationError as exc:
            return cls._wrap_errors(exc, file_override=file)

    def validate(self) -> list[ValidationErrors]:
        """Validate this loaded instance and return errors.

        Re-validates the model's current data against its field types
        and constraints.  Returns an empty list if valid.

        This is intended for the staged workflow where models are first
        loaded with ``validate=False`` and validated later.

        Returns:
            List of ``ValidationErrors`` (empty if valid)
        """
        try:
            type(self).model_validate(self.model_dump())
            return []
        except ValidationError as exc:
            return type(self)._wrap_errors(exc)
