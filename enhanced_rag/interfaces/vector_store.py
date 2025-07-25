"""Abstract interface for vector storage and retrieval."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
import numpy as np


class Document(BaseModel):
    """Document with metadata for vector storage."""
    
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None


class SearchResult(BaseModel):
    """Result of vector similarity search."""
    
    document: Document
    score: float
    rank: int


class VectorStore(ABC):
    """Abstract base class for vector storage and retrieval."""
    
    @abstractmethod
    async def add_documents(
        self, 
        documents: List[Document], 
        embeddings: List[List[float]]
    ) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of documents to add
            embeddings: Corresponding embeddings for each document
            
        Raises:
            VectorStoreError: If adding documents fails
        """
        pass
    
    @abstractmethod
    async def update_documents(
        self, 
        documents: List[Document], 
        embeddings: List[List[float]]
    ) -> None:
        """
        Update existing documents and their embeddings.
        
        Args:
            documents: List of documents to update
            embeddings: Corresponding embeddings for each document
            
        Raises:
            VectorStoreError: If updating documents fails
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Raises:
            VectorStoreError: If deleting documents fails
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform similarity search to find most relevant documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results sorted by relevance
            
        Raises:
            VectorStoreError: If search fails
        """
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by its ID.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            VectorStoreError: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def save_index(self, path: str) -> None:
        """
        Save the vector index to persistent storage.
        
        Args:
            path: Path to save the index
            
        Raises:
            VectorStoreError: If saving fails
        """
        pass
    
    @abstractmethod
    async def load_index(self, path: str) -> None:
        """
        Load the vector index from persistent storage.
        
        Args:
            path: Path to load the index from
            
        Raises:
            VectorStoreError: If loading fails
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing store statistics
        """
        pass


class VectorStoreError(Exception):
    """Exception raised when vector store operations fail."""
    pass