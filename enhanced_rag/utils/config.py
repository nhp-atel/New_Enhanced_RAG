"""Configuration management for the RAG system."""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class Environment(str, Enum):
    """Deployment environments."""
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""
    
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    overlap: int = Field(default=200, ge=0, le=1000)
    strategy: str = Field(default="recursive", pattern="^(recursive|semantic|sentence)$")
    
    @validator('overlap')
    def overlap_must_be_less_than_chunk_size(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('overlap must be less than chunk_size')
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    
    provider: str = Field(default="openai", pattern="^(openai|huggingface|azure)$")
    model_name: str = Field(default="text-embedding-ada-002")
    batch_size: int = Field(default=100, ge=1, le=1000)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class VectorStoreConfig(BaseModel):
    """Configuration for vector storage."""
    
    provider: str = Field(default="faiss", pattern="^(faiss|chroma|weaviate)$")
    index_type: str = Field(default="IndexFlatIP")
    persistence_path: str = Field(default="./data/vector_index")
    save_interval_minutes: int = Field(default=30, ge=1, le=1440)
    backup_enabled: bool = Field(default=True)


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    
    provider: str = Field(default="openai", pattern="^(openai|anthropic|azure)$")
    model_name: str = Field(default="gpt-4-turbo-preview")
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=60, ge=1, le=600)


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""
    
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank_enabled: bool = Field(default=False)
    max_context_length: int = Field(default=8000, ge=1000, le=32000)


class CachingConfig(BaseModel):
    """Configuration for caching."""
    
    enabled: bool = Field(default=True)
    redis_url: Optional[str] = Field(default=None)
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    max_cache_size_mb: int = Field(default=1024, ge=64, le=10240)


class ObservabilityConfig(BaseModel):
    """Configuration for observability."""
    
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="json", pattern="^(json|text)$")
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=8000, ge=1024, le=65535)
    tracing_enabled: bool = Field(default=False)
    correlation_id_header: str = Field(default="X-Correlation-ID")


class RAGConfig(BaseModel):
    """Main RAG system configuration."""
    
    environment: Environment = Field(default=Environment.LOCAL)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    # Runtime settings
    data_directory: str = Field(default="./data")
    temp_directory: str = Field(default="./tmp")
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)


class ConfigManager:
    """Manages configuration loading and environment overrides."""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self._config: Optional[RAGConfig] = None
    
    def load_config(
        self, 
        environment: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> RAGConfig:
        """
        Load configuration with environment-specific overrides.
        
        Args:
            environment: Environment to load config for
            config_overrides: Additional config overrides
            
        Returns:
            Loaded and validated configuration
        """
        # Determine environment
        env = environment or os.getenv("ENVIRONMENT", Environment.LOCAL.value)
        
        # Load base config
        base_config = self._load_config_file("config.yaml")
        
        # Load environment-specific config
        env_config_file = f"{env}.yaml"
        if (self.config_dir / env_config_file).exists():
            env_config = self._load_config_file(env_config_file)
            base_config = self._deep_merge(base_config, env_config)
        
        # Apply environment variable overrides
        base_config = self._apply_env_overrides(base_config)
        
        # Apply manual overrides
        if config_overrides:
            base_config = self._deep_merge(base_config, config_overrides)
        
        # Validate and return
        self._config = RAGConfig(**base_config)
        return self._config
    
    def _load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            if filename == "config.yaml":
                raise FileNotFoundError(f"Base configuration file not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                return yaml.safe_load(f) or {}
            elif filename.endswith('.json'):
                return json.load(f) or {}
            else:
                raise ValueError(f"Unsupported config file format: {filename}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Environment variable mapping
        env_mappings = {
            "RAG_LOG_LEVEL": "observability.log_level",
            "RAG_CHUNK_SIZE": "chunking.chunk_size",
            "RAG_CHUNK_OVERLAP": "chunking.overlap",
            "RAG_EMBEDDING_MODEL": "embedding.model_name",
            "RAG_LLM_MODEL": "llm.model_name",
            "RAG_LLM_TEMPERATURE": "llm.temperature",
            "RAG_TOP_K": "retrieval.top_k",
            "RAG_VECTOR_STORE_PATH": "vector_store.persistence_path",
            "RAG_REDIS_URL": "caching.redis_url",
            "RAG_METRICS_PORT": "observability.metrics_port",
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config, config_path, self._parse_env_value(env_value))
        
        return config
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested dictionary value using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer values
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # String values
        return value
    
    @property
    def config(self) -> RAGConfig:
        """Get the current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config


# Global config manager instance
config_manager = ConfigManager()