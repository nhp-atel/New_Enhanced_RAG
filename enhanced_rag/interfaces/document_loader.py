"""Abstract interface for document loading."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from pathlib import Path


class LoadedDocument(BaseModel):
    """Document loaded from a source."""
    
    content: str
    metadata: Dict[str, Any] = {}
    source: str
    document_id: str


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    async def load_document(self, source: str) -> LoadedDocument:
        """
        Load a single document from a source.
        
        Args:
            source: Path or identifier of the document to load
            
        Returns:
            LoadedDocument containing content and metadata
            
        Raises:
            DocumentLoadError: If loading fails
        """
        pass
    
    @abstractmethod
    async def load_documents(self, sources: List[str]) -> List[LoadedDocument]:
        """
        Load multiple documents from sources.
        
        Args:
            sources: List of paths or identifiers to load
            
        Returns:
            List of LoadedDocument objects
            
        Raises:
            DocumentLoadError: If loading fails
        """
        pass
    
    @abstractmethod
    def supports_source(self, source: str) -> bool:
        """
        Check if this loader supports the given source.
        
        Args:
            source: Source to check support for
            
        Returns:
            True if source is supported, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass


class DocumentLoadError(Exception):
    """Exception raised when document loading fails."""
    pass