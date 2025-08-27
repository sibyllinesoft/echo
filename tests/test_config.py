"""Tests for Echo configuration management."""

import pytest
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from echo.config import EchoConfig


class TestEchoConfig:
    """Test cases for EchoConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EchoConfig()
        
        assert config.min_tokens == 40
        assert config.tau_semantic == 0.83
        assert config.overlap_threshold == 0.75
        assert config.num_hashes == 128
        assert config.num_bands == 16
        assert 'python' in config.supported_languages
        
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_data = {
            'min_tokens': 50,
            'tau_semantic': 0.9,
            'supported_languages': ['python', 'javascript']
        }
        
        # This would be implemented when the from_dict method is added
        # config = EchoConfig.from_dict(config_data)
        # assert config.min_tokens == 50
        # assert config.tau_semantic == 0.9
        
    def test_config_file_save_load(self):
        """Test saving and loading configuration to/from file."""
        config = EchoConfig(min_tokens=60, tau_semantic=0.85)
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
            
        try:
            config.to_file(config_path)
            loaded_config = EchoConfig.from_file(config_path)
            
            assert loaded_config.min_tokens == 60
            assert loaded_config.tau_semantic == 0.85
        finally:
            config_path.unlink(missing_ok=True)
            
    def test_cache_dir_default(self):
        """Test default cache directory creation."""
        config = EchoConfig()
        cache_dir = config.get_cache_dir()
        
        assert cache_dir == Path.home() / '.echo'
        
    def test_models_dir_default(self):
        """Test default models directory creation."""
        config = EchoConfig()
        models_dir = config.get_models_dir()
        
        assert models_dir == Path.home() / '.echo' / 'models'
        
    def test_custom_cache_dir(self):
        """Test custom cache directory."""
        custom_dir = Path('/tmp/echo_test')
        config = EchoConfig(cache_dir=custom_dir)
        
        assert config.get_cache_dir() == custom_dir