"""Abstract interface for embedding generation."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""
    
    embeddings: List[List[float]]
    token_usage: Optional[int] = None
    model_name: str
    dimensions: int


class Embedder(ABC):
    """Abstract base class for embedding generation."""
    
    @abstractmethod
    async def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Optional batch size for processing
            
        Returns:
            EmbeddingResult containing embeddings and metadata
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this embedder."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the embedding model."""
        pass


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    pass