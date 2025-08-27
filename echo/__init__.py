"""
Echo: A duplicate code detection tool for polyglot repositories.

This package provides a comprehensive solution for detecting duplicate code blocks
across multiple programming languages using Tree-sitter parsing, MinHash LSH,
and semantic embeddings.
"""

__version__ = "0.1.0"
__author__ = "Echo Project Contributors"

from .config import EchoConfig
from .parser import extract_blocks
from .scan import scan_changed_files, scan_repository
from .storage import EchoDatabase

__all__ = [
    "scan_repository",
    "scan_changed_files",
    "extract_blocks",
    "EchoDatabase",
    "EchoConfig",
    "__version__",
]
