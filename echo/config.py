"""Configuration management for Echo duplicate code detection."""

from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from .utils import JsonHandler, PathHandler, handle_errors


@dataclass
class EchoConfig:
    """Configuration settings for Echo duplicate code detection."""
    
    # Detection thresholds
    min_tokens: int = 40
    tau_semantic: float = 0.83
    overlap_threshold: float = 0.75
    edit_density_threshold: float = 0.25
    
    # LSH parameters
    num_hashes: int = 128
    num_bands: int = 16
    shingle_size: int = 5
    
    # Performance settings
    max_candidates: int = 50
    max_matches_per_block: int = 5
    min_refactor_score: float = 200.0
    
    # Language support
    supported_languages: List[str] = field(default_factory=lambda: [
        'python', 'typescript', 'javascript'
    ])
    
    # File patterns
    ignore_patterns: List[str] = field(default_factory=lambda: [
        'tests/', '__pycache__/', 'node_modules/', 'vendor/',
        'migrations/', '*.pyc', '*.min.js', '*.bundle.js'
    ])
    
    # Storage paths
    cache_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    
    @classmethod
    @handle_errors("load configuration from file", raise_on_error=True)
    def from_file(cls, path: Path) -> 'EchoConfig':
        """Load configuration from JSON file."""
        data = JsonHandler.read_file(path)
        return cls(**data)
        
    @handle_errors("save configuration to file")
    def to_file(self, path: Path) -> None:
        """Save configuration to JSON file."""
        JsonHandler.write_file(self.__dict__, path)
            
    def get_cache_dir(self) -> Path:
        """Get cache directory, creating default if needed."""
        cache_path = self.cache_dir or (Path.home() / '.echo')
        PathHandler.ensure_directory(cache_path)
        return cache_path
        
    def get_models_dir(self) -> Path:
        """Get models directory, creating default if needed."""
        models_path = self.models_dir or (self.get_cache_dir() / 'models')
        PathHandler.ensure_directory(models_path)
        return models_path