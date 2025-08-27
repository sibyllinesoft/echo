"""Common utilities for Echo duplicate code detection.

This module provides shared utilities to reduce code duplication and improve
consistency across the Echo codebase.
"""

import json
import logging
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

# Type variable for generic error handling
T = TypeVar("T")

logger = logging.getLogger(__name__)


class EchoError(Exception):
    """Base exception for Echo-specific errors."""

    pass


class ConfigurationError(EchoError):
    """Raised when configuration is invalid or missing."""

    pass


class StorageError(EchoError):
    """Raised when storage operations fail."""

    pass


class ParsingError(EchoError):
    """Raised when code parsing fails."""

    pass


def handle_errors(
    operation_name: str,
    default_return: T = None,
    raise_on_error: bool = False,
    log_level: int = logging.ERROR,
) -> Callable:
    """Decorator to standardize error handling across the codebase.

    Args:
        operation_name: Human-readable name for the operation
        default_return: Value to return if operation fails
        raise_on_error: Whether to re-raise exceptions after logging
        log_level: Logging level for error messages
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(log_level, f"Failed to {operation_name}: {e}")
                if raise_on_error:
                    raise
                return default_return

        return wrapper

    return decorator


@contextmanager
def database_session_handler(session) -> Iterator:
    """Standard database session handling with automatic cleanup.

    Provides consistent session management with proper error handling
    and cleanup across all database operations.

    Args:
        session: SQLAlchemy session object
    """
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database operation failed: {e}")
        # Try to import SQLAlchemyError for specific error handling
        try:
            from sqlalchemy.exc import SQLAlchemyError

            if isinstance(e, SQLAlchemyError):
                raise StorageError(f"Database operation failed: {e}") from e
        except ImportError:
            pass
        raise
    finally:
        session.close()


class JsonHandler:
    """Centralized JSON serialization and deserialization with error handling."""

    @staticmethod
    @handle_errors("serialize to JSON", default_return="{}")
    def serialize(obj: Any, indent: int = 2, ensure_ascii: bool = False) -> str:
        """Serialize object to JSON string with consistent formatting."""
        return json.dumps(obj, indent=indent, default=str, ensure_ascii=ensure_ascii)

    @staticmethod
    @handle_errors("parse JSON", default_return={})
    def deserialize(json_str: str) -> Dict[str, Any]:
        """Deserialize JSON string to dictionary."""
        if not json_str or not json_str.strip():
            return {}
        return json.loads(json_str)

    @staticmethod
    @handle_errors("write JSON file")
    def write_file(obj: Any, file_path: Path, indent: int = 2) -> bool:
        """Write object to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, default=str, ensure_ascii=False)
        return True

    @staticmethod
    @handle_errors("read JSON file", default_return={})
    def read_file(file_path: Path) -> Dict[str, Any]:
        """Read JSON file to dictionary."""
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


class PathHandler:
    """Centralized path operations with validation and error handling."""

    @staticmethod
    def normalize_path(path: Path) -> str:
        """Normalize path to consistent string format."""
        return path.as_posix()

    @staticmethod
    @handle_errors("validate path", default_return=False)
    def is_valid_file(path: Path) -> bool:
        """Check if path points to a valid, readable file."""
        return path.exists() and path.is_file() and path.stat().st_size > 0

    @staticmethod
    @handle_errors("create directory")
    def ensure_directory(path: Path) -> bool:
        """Ensure directory exists, creating if necessary."""
        path.mkdir(parents=True, exist_ok=True)
        return True

    @staticmethod
    def get_relative_path(path: Path, base: Path) -> str:
        """Get relative path string from base directory."""
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            return path.as_posix()


class ProgressReporter:
    """Standardized progress reporting across operations."""

    def __init__(self, operation_name: str, total_items: Optional[int] = None):
        self.operation_name = operation_name
        self.total_items = total_items
        self.processed_items = 0
        self.logger = logging.getLogger(f"{__name__}.progress")

    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress counter and log if significant milestone."""
        self.processed_items += increment

        # Log at intervals or when message provided
        if message or (self.processed_items % 100 == 0):
            if self.total_items:
                percentage = (self.processed_items / self.total_items) * 100
                log_msg = f"{self.operation_name}: {self.processed_items}/{self.total_items} ({percentage:.1f}%)"
            else:
                log_msg = f"{self.operation_name}: {self.processed_items} processed"

            if message:
                log_msg += f" - {message}"

            self.logger.info(log_msg)

    def complete(self, final_message: Optional[str] = None):
        """Mark operation as complete."""
        if final_message:
            self.logger.info(f"{self.operation_name} complete: {final_message}")
        else:
            self.logger.info(
                f"{self.operation_name} complete: {self.processed_items} items processed"
            )


def validate_config(config, required_fields: Dict[str, type]) -> bool:
    """Validate configuration object has required fields of correct types."""
    for field_name, expected_type in required_fields.items():
        if not hasattr(config, field_name):
            raise ConfigurationError(
                f"Missing required configuration field: {field_name}"
            )

        field_value = getattr(config, field_name)
        if not isinstance(field_value, expected_type):
            raise ConfigurationError(
                f"Configuration field {field_name} must be {expected_type.__name__}, "
                f"got {type(field_value).__name__}"
            )

    return True


def safe_file_operation(operation_name: str):
    """Decorator for file operations with consistent error handling."""

    def decorator(func):
        @wraps(func)
        @handle_errors(f"perform file operation: {operation_name}")
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
