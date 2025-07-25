"""Text chunking strategies for document processing."""

import re
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    
    content: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        return len(self.content)


class TextChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
    
    @abstractmethod
    def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of text chunks
        """
        pass
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate a unique ID for a chunk."""
        import hashlib
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"chunk_{index}_{content_hash}"


class RecursiveTextChunker(TextChunker):
    """
    Recursive text chunker that tries to split on semantic boundaries.
    
    Attempts to split text on paragraph boundaries, sentence boundaries,
    and finally character boundaries as fallbacks.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        super().__init__(chunk_size, overlap)
        
        # Separators in order of preference
        self.separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation sentences
            "? ",    # Question sentences
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Character-level split
        ]
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Split text recursively using semantic boundaries."""
        if not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Start recursive splitting
        text_chunks = self._recursive_split(text, self.separators)
        
        # Combine small chunks and split large ones
        final_chunks = self._merge_and_split_chunks(text_chunks)
        
        # Create TextChunk objects
        current_pos = 0
        for i, chunk_text in enumerate(final_chunks):
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            chunk = TextChunk(
                content=chunk_text.strip(),
                start_index=start_pos,
                end_index=end_pos,
                chunk_id=self._generate_chunk_id(chunk_text, i),
                metadata={**metadata, "chunk_index": i, "strategy": "recursive"}
            )
            chunks.append(chunk)
            current_pos = end_pos
        
        logger.info(f"Created {len(chunks)} chunks using recursive strategy")
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if not separator:
            # Character-level split
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        # Split by current separator
        splits = text.split(separator)
        
        # If we only have one split, try the next separator
        if len(splits) == 1:
            return self._recursive_split(text, remaining_separators)
        
        # Reconstruct text with separators and check chunk sizes
        good_splits = []
        current_chunk = ""
        
        for i, split in enumerate(splits):
            if i > 0:
                test_chunk = current_chunk + separator + split
            else:
                test_chunk = split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    good_splits.append(current_chunk)
                
                # If single split is too large, recursively split it
                if len(split) > self.chunk_size:
                    sub_splits = self._recursive_split(split, remaining_separators)
                    good_splits.extend(sub_splits)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        if current_chunk:
            good_splits.append(current_chunk)
        
        return good_splits
    
    def _merge_and_split_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks and ensure proper overlap."""
        if not chunks:
            return []
        
        result = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Try to merge with next chunks if current is too small
            while (len(current_chunk) < self.chunk_size * 0.5 and 
                   i + 1 < len(chunks)):
                next_chunk = chunks[i + 1]
                if len(current_chunk) + len(next_chunk) <= self.chunk_size:
                    current_chunk += " " + next_chunk
                    i += 1
                else:
                    break
            
            result.append(current_chunk)
            i += 1
        
        # Add overlap between chunks
        if self.overlap > 0 and len(result) > 1:
            result = self._add_overlap(result)
        
        return result
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if not chunks or len(chunks) < 2:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # Get overlap text from previous chunk
            overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) > self.overlap else prev_chunk
            
            # Add overlap to current chunk
            overlapped_chunk = overlap_text + " " + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks


class SentenceTextChunker(TextChunker):
    """Chunks text by sentence boundaries while respecting size limits."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        super().__init__(chunk_size, overlap)
        
        # Pattern for sentence endings
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Split text by sentences while respecting chunk size."""
        if not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Split into sentences
        sentences = self.sentence_pattern.split(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
                current_sentences.append(sentence)
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk
                if len(sentence) > self.chunk_size:
                    # Split long sentence
                    sub_chunks = self._split_long_sentence(sentence)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
                    current_sentences = [current_chunk]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Create TextChunk objects
        result_chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunks):
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            chunk = TextChunk(
                content=chunk_text.strip(),
                start_index=start_pos,
                end_index=end_pos,
                chunk_id=self._generate_chunk_id(chunk_text, i),
                metadata={**metadata, "chunk_index": i, "strategy": "sentence"}
            )
            result_chunks.append(chunk)
            current_pos = end_pos
        
        logger.info(f"Created {len(result_chunks)} chunks using sentence strategy")
        return result_chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a sentence that's too long for chunk size."""
        words = sentence.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class SemanticTextChunker(TextChunker):
    """
    Semantic chunker that uses embedding similarity to group related content.
    Note: This is a placeholder implementation. Full semantic chunking would
    require embedding models and similarity calculations.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        super().__init__(chunk_size, overlap)
        logger.warning("SemanticTextChunker is a placeholder. Using recursive chunking.")
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Placeholder semantic chunking - falls back to recursive."""
        # For now, use recursive chunking as fallback
        recursive_chunker = RecursiveTextChunker(self.chunk_size, self.overlap)
        chunks = recursive_chunker.chunk_text(text, metadata)
        
        # Update metadata to indicate semantic strategy was requested
        for chunk in chunks:
            chunk.metadata["strategy"] = "semantic"
            chunk.metadata["fallback"] = "recursive"
        
        return chunks


class ChunkerFactory:
    """Factory for creating text chunkers."""
    
    @staticmethod
    def create_chunker(
        strategy: ChunkingStrategy,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> TextChunker:
        """Create a text chunker based on strategy."""
        
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveTextChunker(chunk_size, overlap)
        elif strategy == ChunkingStrategy.SENTENCE:
            return SentenceTextChunker(chunk_size, overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticTextChunker(chunk_size, overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


# Convenience function
def chunk_text(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 1000,
    overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None
) -> List[TextChunk]:
    """
    Convenience function to chunk text with specified strategy.
    
    Args:
        text: Text to chunk
        strategy: Chunking strategy to use
        chunk_size: Maximum size of each chunk
        overlap: Overlap between consecutive chunks
        metadata: Optional metadata to include with chunks
        
    Returns:
        List of text chunks
    """
    chunker = ChunkerFactory.create_chunker(strategy, chunk_size, overlap)
    return chunker.chunk_text(text, metadata)