# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""BaseConfig — Pydantic model with automatic IO strategy registration."""

from typing import ClassVar, Optional, Type, TypeVar, Union, cast
from pathlib import Path

from pydantic import BaseModel, ValidationError

from neofoam.io.validation_types import (
    IOMetadata,
    ReadingStrategy,
    WritingStrategy,
    ValidationErrors,
)

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base class for configurations with reading/writing strategies.

    Subclasses decorated with ``@IOStrategy`` get an ``io_config``
    class variable of type :class:`IOMetadata`.
    """

    io_config: ClassVar[Optional[IOMetadata]] = None

    @classmethod
    def _get_io(cls) -> IOMetadata:
        """Return ``io_config`` or raise if not registered."""
        if cls.io_config is None:
            raise ValueError(f"No IO strategy registered for {cls.__name__}")
        return cls.io_config

    @classmethod
    def get_default_path(cls, case_dir: Union[Path, str] = ".") -> Path:
        """Get default file path from ``io_config.file``."""
        return Path(case_dir) / cls._get_io().file

    @classmethod
    def get_reading_strategy(cls) -> ReadingStrategy:
        """Get the reading strategy for this config class."""
        return cls._get_io().reader

    @classmethod
    def get_writing_strategy(cls) -> WritingStrategy:
        """Get the writing strategy for this config class."""
        return cls._get_io().writer

    @classmethod
    def set_reading_strategy(cls, strategy: ReadingStrategy) -> None:
        """Set custom reading strategy for this config class."""
        cls._get_io().reader = strategy

    @classmethod
    def set_writing_strategy(cls, strategy: WritingStrategy) -> None:
        """Set custom writing strategy for this config class."""
        cls._get_io().writer = strategy

    @classmethod
    def load(
        cls: Type[T],
        *,
        case_dir: Union[Path, str] = ".",
        file: Optional[str] = None,
        validate: bool = True,
        encoding: str = "utf-8",
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
        data = reading_strategy.read(cls, path, encoding=encoding)

        if validate:
            return cls.model_validate(data)

        return cast(T, cls.model_construct(**data))

    @property
    def configs(self) -> list["BaseConfig"]:
        """Convenience property to get all config models together."""
        return [self]

    @property
    def file_name(self) -> str:
        """Get the file name for this configuration.

        Returns:
            Relative path to the configuration file

        Example:
            >>> config = SolverConfig(...)
            >>> config.file_name
            'solver_config.yaml'
        """
        io = type(self).io_config
        return io.file if io else type(self).__name__

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
        io = type(self).io_config
        return io.subdict if io else None

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
        *,
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
        writing_strategy.write(self, path, encoding=encoding)

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
        file_name = file_override or (
            cls.io_config.file if cls.io_config else cls.__name__
        )

        # Resolve subdict from io_config
        subdict = cls.io_config.subdict if cls.io_config else None

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
        *,
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
        file_name = file or (cls.io_config.file if cls.io_config else cls.__name__)

        try:
            cls.load(case_dir=case_dir, encoding=encoding, validate=True, file=file)
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

    def check_validation(self) -> list[ValidationErrors]:
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
