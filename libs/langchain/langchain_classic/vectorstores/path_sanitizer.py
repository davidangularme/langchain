import os
import re
from pathlib import Path
from typing import Union

class PathSanitizer:
    """
    A wrapper to sanitize and validate file paths to prevent path traversal attacks.
    All provided paths are resolved to their absolute form, checked against a base directory,
    and normalized to prevent directory traversal.
    """

    def __init__(self, base_directory: Union[str, Path] = None):
        """
        Initialize the sanitizer with a base directory.
        If no base directory is provided, the current working directory is used.
        """
        if base_directory is None:
            base_directory = Path.cwd()
        self._base_directory = Path(base_directory).resolve()

    def sanitize(self, user_path: Union[str, Path]) -> Path:
        """
        Sanitize and validate a user-supplied path.
        Raises ValueError if the path is malicious or cannot be safely resolved.
        """
        if not isinstance(user_path, (str, Path)):
            raise ValueError("Path must be a string or Path object")

        # Convert to string and strip whitespace/null bytes
        path_str = str(user_path).strip()
        if not path_str:
            raise ValueError("Path cannot be empty")

        # Remove null bytes and control characters
        if '\\x00' in path_str or any(ord(c) < 32 for c in path_str):
            raise ValueError("Path contains invalid characters")

        # Reject obvious traversal attempts (e.g., absolute paths, `..` at start)
        if path_str.startswith('/') or path_str.startswith('\\\\'):
            raise ValueError("Absolute paths are not allowed")
        if '..' in Path(path_str).parts:
            raise ValueError("Path traversal detected")

        # Construct full path by joining with base directory
        full_path = (self._base_directory / path_str).resolve()

        # Ensure the resolved path is within the base directory
        try:
            full_path.relative_to(self._base_directory)
        except ValueError:
            raise ValueError("Resolved path is outside the allowed base directory")

        # Check for symlinks that could escape
        if full_path.is_symlink():
            real_path = full_path.resolve()
            try:
                real_path.relative_to(self._base_directory)
            except ValueError:
                raise ValueError("Symlink points outside the allowed base directory")

        return full_path

    def sanitize_read(self, user_path: Union[str, Path]) -> Path:
        """Sanitize path for read operations (file must exist)."""
        full_path = self.sanitize(user_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Path does not exist: {full_path}")
        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {full_path}")
        # Ensure we can read
        if not os.access(full_path, os.R_OK):
            raise PermissionError(f"No read permission: {full_path}")
        return full_path

    def sanitize_write(self, user_path: Union[str, Path]) -> Path:
        """Sanitize path for write operations (parent directory must exist)."""
        full_path = self.sanitize(user_path)
        parent = full_path.parent
        if not parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {parent}")
        # Ensure we can write to parent
        if not os.access(parent, os.W_OK) if parent.is_dir() else os.access(parent.parent, os.W_OK):
            raise PermissionError(f"No write permission: {full_path}")
        return full_path

    def sanitize_delete(self, user_path: Union[str, Path]) -> Path:
        """Sanitize path for delete operations (file must exist and be writable)."""
        full_path = self.sanitize_read(user_path)  # Ensure exists and readable
        if not os.access(full_path.parent, os.W_OK):
            raise PermissionError(f"No delete permission: {full_path}")
        return full_path
