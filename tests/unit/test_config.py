"""Unit tests for configuration management."""

import json
import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from enhanced_rag.utils.config import (
    ConfigManager,
    RAGConfig,
    ChunkingConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    LLMConfig,
    RetrievalConfig,
    CachingConfig,
    ObservabilityConfig,
    Environment
)


class TestConfigModels:
    """Test configuration model validation."""
    
    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # Valid config
        config = ChunkingConfig(chunk_size=1000, overlap=200)
        assert config.chunk_size == 1000
        assert config.overlap == 200
        
        # Invalid overlap
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=500, overlap=600)
    
    def test_embedding_config_validation(self):
        """Test embedding configuration validation."""
        config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-ada-002",
            batch_size=100
        )
        assert config.provider == "openai"
        assert config.batch_size == 100
    
    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        config = LLMConfig(
            provider="openai",
            model_name="gpt-4-turbo-preview",
            temperature=0.1
        )
        assert config.provider == "openai"
        assert config.temperature == 0.1
    
    def test_retrieval_config_validation(self):
        """Test retrieval configuration validation."""
        config = RetrievalConfig(top_k=5, score_threshold=0.7)
        assert config.top_k == 5
        assert config.score_threshold == 0.7
    
    def test_observability_config_validation(self):
        """Test observability configuration validation."""
        config = ObservabilityConfig(
            log_level="INFO",
            log_format="json",
            metrics_enabled=True
        )
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.metrics_enabled is True


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def test_init(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager("/tmp/config")
        assert config_manager.config_dir == Path("/tmp/config")
        assert config_manager._config is None
    
    def test_load_config_file_yaml(self):
        """Test loading YAML configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Create test config file
            config_data = {
                "environment": "local",
                "chunking": {"chunk_size": 1500, "overlap": 300}
            }
            
            config_file = Path(temp_dir) / "test.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load config
            loaded_config = config_manager._load_config_file("test.yaml")
            
            assert loaded_config["environment"] == "local"
            assert loaded_config["chunking"]["chunk_size"] == 1500
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Non-existent file should return empty dict
            result = config_manager._load_config_file("nonexistent.yaml")
            assert result == {}
    
    def test_load_config_base_not_found(self):
        """Test loading when base config file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            with pytest.raises(FileNotFoundError):
                config_manager.load_config()
    
    def test_deep_merge(self):
        """Test deep merging of configuration dictionaries."""
        config_manager = ConfigManager()
        
        base = {
            "chunking": {"chunk_size": 1000, "overlap": 200},
            "embedding": {"provider": "openai"}
        }
        
        override = {
            "chunking": {"chunk_size": 1500},
            "llm": {"provider": "anthropic"}
        }
        
        result = config_manager._deep_merge(base, override)
        
        assert result["chunking"]["chunk_size"] == 1500  # Overridden
        assert result["chunking"]["overlap"] == 200      # Preserved
        assert result["embedding"]["provider"] == "openai"  # Preserved
        assert result["llm"]["provider"] == "anthropic"  # Added
    
    def test_parse_env_value(self):
        """Test environment variable value parsing."""
        config_manager = ConfigManager()
        
        # Boolean values
        assert config_manager._parse_env_value("true") is True
        assert config_manager._parse_env_value("false") is False
        assert config_manager._parse_env_value("True") is True
        assert config_manager._parse_env_value("FALSE") is False
        
        # Integer values
        assert config_manager._parse_env_value("42") == 42
        assert config_manager._parse_env_value("-10") == -10
        
        # Float values
        assert config_manager._parse_env_value("3.14") == 3.14
        assert config_manager._parse_env_value("-0.5") == -0.5
        
        # String values
        assert config_manager._parse_env_value("hello") == "hello"
        assert config_manager._parse_env_value("123abc") == "123abc"
    
    def test_set_nested_value(self):
        """Test setting nested dictionary values."""
        config_manager = ConfigManager()
        
        data = {}
        config_manager._set_nested_value(data, "chunking.chunk_size", 1500)
        
        assert data["chunking"]["chunk_size"] == 1500
        
        # Test setting deeper nesting
        config_manager._set_nested_value(data, "llm.config.temperature", 0.7)
        assert data["llm"]["config"]["temperature"] == 0.7
    
    @patch.dict(os.environ, {
        "RAG_LOG_LEVEL": "DEBUG",
        "RAG_CHUNK_SIZE": "1500",
        "RAG_LLM_TEMPERATURE": "0.5",
        "RAG_TOP_K": "10"
    })
    def test_apply_env_overrides(self):
        """Test applying environment variable overrides."""
        config_manager = ConfigManager()
        
        base_config = {
            "observability": {"log_level": "INFO"},
            "chunking": {"chunk_size": 1000},
            "llm": {"temperature": 0.1},
            "retrieval": {"top_k": 5}
        }
        
        result = config_manager._apply_env_overrides(base_config)
        
        assert result["observability"]["log_level"] == "DEBUG"
        assert result["chunking"]["chunk_size"] == 1500
        assert result["llm"]["temperature"] == 0.5
        assert result["retrieval"]["top_k"] == 10
    
    def test_load_config_with_environment_override(self):
        """Test loading config with environment-specific overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Create base config
            base_config = {
                "environment": "local",
                "chunking": {"chunk_size": 1000},
                "embedding": {"provider": "openai"}
            }
            
            with open(Path(temp_dir) / "config.yaml", 'w') as f:
                yaml.dump(base_config, f)
            
            # Create environment-specific config
            env_config = {
                "chunking": {"chunk_size": 1500},
                "llm": {"provider": "anthropic"}
            }
            
            with open(Path(temp_dir) / "staging.yaml", 'w') as f:
                yaml.dump(env_config, f)
            
            # Load with environment override
            result = config_manager.load_config("staging")
            
            assert result.chunking.chunk_size == 1500  # From staging override
            assert result.embedding.provider == "openai"  # From base config
    
    def test_config_property_before_load(self):
        """Test accessing config property before loading."""
        config_manager = ConfigManager()
        
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            config_manager.config
    
    def test_config_property_after_load(self):
        """Test accessing config property after loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Create minimal config
            base_config = {"environment": "local"}
            
            with open(Path(temp_dir) / "config.yaml", 'w') as f:
                yaml.dump(base_config, f)
            
            # Load config
            loaded_config = config_manager.load_config()
            
            # Property should return the same config
            assert config_manager.config == loaded_config
    
    def test_load_config_with_overrides(self):
        """Test loading config with manual overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Create base config
            base_config = {
                "environment": "local",
                "chunking": {"chunk_size": 1000}
            }
            
            with open(Path(temp_dir) / "config.yaml", 'w') as f:
                yaml.dump(base_config, f)
            
            # Load with manual overrides
            overrides = {
                "chunking": {"chunk_size": 2000, "overlap": 400},
                "llm": {"temperature": 0.8}
            }
            
            result = config_manager.load_config(config_overrides=overrides)
            
            assert result.chunking.chunk_size == 2000
            assert result.chunking.overlap == 400
    
    def test_load_config_file_json(self):
        """Test loading JSON config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Create JSON config file
            json_config = {"chunking": {"chunk_size": 2000}, "llm": {"model": "gpt-4"}}
            json_file = Path(temp_dir) / "config.json"
            
            with open(json_file, 'w') as f:
                json.dump(json_config, f)
            
            # Load the JSON config
            loaded_config = config_manager._load_config_file("config.json")
            
            assert loaded_config["chunking"]["chunk_size"] == 2000
            assert loaded_config["llm"]["model"] == "gpt-4"
    
    def test_unsupported_config_file_format(self):
        """Test loading unsupported config file format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            
            # Create unsupported file
            unsupported_file = Path(temp_dir) / "config.txt"
            unsupported_file.write_text("some content")
            
            with pytest.raises(ValueError, match="Unsupported config file format"):
                config_manager._load_config_file("config.txt")


class TestGlobalConfigManager:
    """Test global config manager instance."""
    
    def test_global_config_manager_import(self):
        """Test importing global config manager."""
        from enhanced_rag.utils.config import config_manager
        
        assert isinstance(config_manager, ConfigManager)
        assert config_manager.config_dir == Path("./config")


class TestEnvironmentEnum:
    """Test Environment enum."""
    
    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.LOCAL == "local"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"