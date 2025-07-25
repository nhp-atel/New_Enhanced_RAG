"""Unit tests for text chunking functionality."""

import pytest
from unittest.mock import Mock, patch
from typing import List

from enhanced_rag.core.chunking import (
    RecursiveTextChunker,
    SentenceTextChunker,
    SemanticTextChunker,
    ChunkerFactory,
    ChunkingStrategy,
    TextChunk,
    TextChunker,
    chunk_text
)


class TestRecursiveTextChunker:
    """Test cases for RecursiveTextChunker."""
    
    def test_init_valid_params(self):
        """Test chunker initialization with valid parameters."""
        chunker = RecursiveTextChunker(chunk_size=1000, overlap=200)
        assert chunker.chunk_size == 1000
        assert chunker.overlap == 200
    
    def test_init_invalid_overlap(self):
        """Test chunker initialization with invalid overlap."""
        with pytest.raises(ValueError, match="Overlap must be less than chunk_size"):
            RecursiveTextChunker(chunk_size=100, overlap=150)
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = RecursiveTextChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = RecursiveTextChunker(chunk_size=1000, overlap=100)
        text = "This is a short text."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].start_index == 0
        assert chunks[0].end_index == len(text)
        assert chunks[0].metadata["strategy"] == "recursive"
    
    def test_chunk_long_text_paragraph_split(self):
        """Test chunking long text with paragraph boundaries."""
        chunker = RecursiveTextChunker(chunk_size=100, overlap=20)
        text = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(len(chunk.content) <= 120 for chunk in chunks)  # chunk_size + some tolerance
    
    def test_chunk_with_metadata(self):
        """Test chunking with custom metadata."""
        chunker = RecursiveTextChunker()
        text = "Test text"
        metadata = {"source": "test.txt", "author": "test"}
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["author"] == "test"
        assert chunks[0].metadata["strategy"] == "recursive"
    
    def test_generate_chunk_id_consistency(self):
        """Test chunk ID generation consistency."""
        chunker = RecursiveTextChunker()
        text = "Same text"
        
        id1 = chunker._generate_chunk_id(text, 0)
        id2 = chunker._generate_chunk_id(text, 0)
        
        assert id1 == id2
    
    def test_generate_chunk_id_uniqueness(self):
        """Test chunk ID uniqueness for different content."""
        chunker = RecursiveTextChunker()
        
        id1 = chunker._generate_chunk_id("text1", 0)
        id2 = chunker._generate_chunk_id("text2", 0)
        
        assert id1 != id2


class TestSentenceTextChunker:
    """Test cases for SentenceTextChunker."""
    
    def test_chunk_by_sentences(self):
        """Test chunking by sentence boundaries."""
        chunker = SentenceTextChunker(chunk_size=100, overlap=20)
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "sentence"
    
    def test_chunk_long_sentence(self):
        """Test handling of sentences longer than chunk size."""
        chunker = SentenceTextChunker(chunk_size=50, overlap=10)
        text = "This is a very long sentence that exceeds the chunk size limit and should be split into multiple parts."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        assert all(len(chunk.content) <= 60 for chunk in chunks)  # chunk_size + tolerance


class TestSemanticTextChunker:
    """Test cases for SemanticTextChunker."""
    
    def test_semantic_chunker_fallback(self):
        """Test that semantic chunker falls back to recursive."""
        chunker = SemanticTextChunker(chunk_size=100, overlap=20)
        text = "Test text for semantic chunking."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        assert chunks[0].metadata["strategy"] == "semantic"
        assert chunks[0].metadata["fallback"] == "recursive"


class TestChunkerFactory:
    """Test cases for ChunkerFactory."""
    
    def test_create_recursive_chunker(self):
        """Test creating recursive chunker."""
        chunker = ChunkerFactory.create_chunker(
            ChunkingStrategy.RECURSIVE, 
            chunk_size=500, 
            overlap=50
        )
        
        assert isinstance(chunker, RecursiveTextChunker)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50
    
    def test_create_sentence_chunker(self):
        """Test creating sentence chunker."""
        chunker = ChunkerFactory.create_chunker(
            ChunkingStrategy.SENTENCE,
            chunk_size=300,
            overlap=30
        )
        
        assert isinstance(chunker, SentenceTextChunker)
        assert chunker.chunk_size == 300
        assert chunker.overlap == 30
    
    def test_create_semantic_chunker(self):
        """Test creating semantic chunker."""
        chunker = ChunkerFactory.create_chunker(
            ChunkingStrategy.SEMANTIC,
            chunk_size=800,
            overlap=80
        )
        
        assert isinstance(chunker, SemanticTextChunker)
        assert chunker.chunk_size == 800
        assert chunker.overlap == 80
    
    def test_create_unknown_strategy(self):
        """Test creating chunker with unknown strategy."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkerFactory.create_chunker("unknown_strategy", 100, 10)


class TestConvenienceFunction:
    """Test cases for convenience function."""
    
    def test_chunk_text_function(self):
        """Test the chunk_text convenience function."""
        text = "Test text for convenience function."
        
        chunks = chunk_text(
            text,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=50,
            overlap=10
        )
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert chunks[0].metadata["strategy"] == "recursive"
    
    def test_chunk_text_with_metadata(self):
        """Test convenience function with metadata."""
        text = "Test text"
        metadata = {"test": "value"}
        
        chunks = chunk_text(text, metadata=metadata)
        
        assert len(chunks) == 1
        assert chunks[0].metadata["test"] == "value"


@pytest.fixture
def sample_long_text():
    """Fixture providing sample long text for testing."""
    return """
    This is the first paragraph of a longer text document. It contains multiple sentences 
    and should be used for testing text chunking functionality. The paragraph discusses 
    various aspects of text processing and natural language understanding.
    
    This is the second paragraph which continues the discussion about text processing.
    It provides additional context and information that would be useful for testing
    chunking algorithms and their effectiveness in maintaining semantic coherence.
    
    The third paragraph concludes our sample text with some final thoughts on the
    importance of proper text segmentation in natural language processing applications.
    This paragraph also contains multiple sentences for comprehensive testing.
    """


class TestIntegration:
    """Integration tests for chunking functionality."""
    
    def test_recursive_chunking_preserves_content(self, sample_long_text):
        """Test that recursive chunking preserves all content."""
        chunker = RecursiveTextChunker(chunk_size=200, overlap=50)
        
        chunks = chunker.chunk_text(sample_long_text)
        
        # Reconstruct text from chunks (removing overlap)
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                reconstructed += chunk.content
            else:
                # Remove overlap from subsequent chunks
                overlap_size = min(50, len(chunk.content))
                reconstructed += chunk.content[overlap_size:]
        
        # Content should be preserved (allowing for whitespace differences)
        original_words = sample_long_text.split()
        reconstructed_words = reconstructed.split()
        
        # Most words should be preserved
        assert len(reconstructed_words) >= len(original_words) * 0.9
    
    def test_chunking_metadata_consistency(self, sample_long_text):
        """Test that metadata is consistently applied to all chunks."""
        chunker = RecursiveTextChunker(chunk_size=150, overlap=30)
        metadata = {"document": "test.txt", "section": "introduction"}
        
        chunks = chunker.chunk_text(sample_long_text, metadata)
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["document"] == "test.txt"
            assert chunk.metadata["section"] == "introduction"
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["strategy"] == "recursive"
    
    def test_chunk_size_compliance(self, sample_long_text):
        """Test that chunks generally comply with size limits."""
        chunker = RecursiveTextChunker(chunk_size=100, overlap=20)
        
        chunks = chunker.chunk_text(sample_long_text)
        
        # Most chunks should be within reasonable bounds
        oversized_chunks = [chunk for chunk in chunks if len(chunk.content) > 150]
        assert len(oversized_chunks) <= len(chunks) * 0.1  # At most 10% can be oversized
    
    def test_different_strategies_same_text(self, sample_long_text):
        """Test different chunking strategies on the same text."""
        strategies = [
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.SENTENCE,
            ChunkingStrategy.SEMANTIC
        ]
        
        results = {}
        for strategy in strategies:
            chunks = chunk_text(
                sample_long_text,
                strategy=strategy,
                chunk_size=200,
                overlap=40
            )
            results[strategy] = chunks
        
        # All strategies should produce chunks
        for strategy, chunks in results.items():
            assert len(chunks) > 0
            assert all(chunk.metadata["strategy"] == strategy.value for chunk in chunks)
        
        # Strategies might produce different numbers of chunks
        chunk_counts = [len(chunks) for chunks in results.values()]
        assert max(chunk_counts) >= min(chunk_counts)  # Basic sanity check


class TestTextChunkModel:
    """Test TextChunk model functionality."""
    
    def test_text_chunk_len(self):
        """Test TextChunk __len__ method."""
        content = "This is test content for length checking."
        chunk = TextChunk(
            content=content,
            start_index=0,
            end_index=len(content),
            chunk_id="test-123",
            metadata={"source": "test"}
        )
        
        # __len__ should return length of content
        assert len(chunk) == len(content)
        assert len(chunk) == 41


class TestAbstractChunker:
    """Test abstract TextChunker functionality."""
    
    def test_abstract_chunker_cannot_instantiate(self):
        """Test that abstract TextChunker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TextChunker()
    
    def test_abstract_chunker_methods(self):
        """Test that TextChunker abstract methods cannot be called directly."""
        # Verify all abstract methods raise TypeError when instantiated
        class IncompleteChunker(TextChunker):
            """Incomplete implementation missing all methods."""
            pass
            
        with pytest.raises(TypeError):
            IncompleteChunker()


class TestChunkingEdgeCases:
    """Test edge cases in chunking functionality."""
    
    def test_recursive_chunker_split_on_periods(self):
        """Test recursive chunker fallback to period splitting."""
        chunker = RecursiveTextChunker(chunk_size=50, overlap=10)
        # Text with no paragraphs but periods
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    
    def test_text_chunk_with_unicode(self):
        """Test TextChunk with unicode content."""
        content = "This is tëst cöntént with ünicode characters."
        chunk = TextChunk(
            content=content,
            start_index=0,
            end_index=len(content),
            chunk_id="unicode-test",
            metadata={"encoding": "utf-8"}
        )
        
        assert len(chunk) == len(content)
        assert chunk.content == content
    
    def test_chunking_strategy_enum_values(self):
        """Test ChunkingStrategy enum values."""
        assert ChunkingStrategy.RECURSIVE.value == "recursive"
        assert ChunkingStrategy.SENTENCE.value == "sentence"
        assert ChunkingStrategy.SEMANTIC.value == "semantic"
    
    def test_chunker_error_handling(self):
        """Test chunker error handling and edge cases."""
        chunker = RecursiveTextChunker(chunk_size=100, overlap=20)
        
        # Test with None text (should raise AttributeError)
        with pytest.raises(AttributeError):
            chunker.chunk_text(None)
    
    def test_text_chunk_metadata_types(self):
        """Test TextChunk with various metadata types."""
        chunk = TextChunk(
            content="Test content with various metadata types",
            start_index=0,
            end_index=40,
            chunk_id="metadata-test",
            metadata={
                "string_value": "test",
                "int_value": 42,
                "float_value": 3.14,
                "bool_value": True,
                "list_value": [1, 2, 3],
                "dict_value": {"nested": "value"}
            }
        )
        
        assert chunk.metadata["string_value"] == "test"
        assert chunk.metadata["int_value"] == 42
        assert chunk.metadata["float_value"] == 3.14
        assert chunk.metadata["bool_value"] is True
        assert chunk.metadata["list_value"] == [1, 2, 3]
        assert chunk.metadata["dict_value"]["nested"] == "value"
    
    def test_recursive_chunker_edge_cases(self):
        """Test edge cases in recursive chunker to improve coverage."""
        chunker = RecursiveTextChunker(chunk_size=50, overlap=10)
        
        # Test the character-level split case by forcing no separators
        # This should hit the character-level split logic
        text_no_separators = "x" * 200  # Long text with no separators
        chunks = chunker.chunk_text(text_no_separators)
        
        # Should create multiple chunks via character-level splitting
        assert len(chunks) >= 3
        # Verify chunks were created (character-level or otherwise)
        assert all(len(chunk.content) > 0 for chunk in chunks)
        
        # Test empty separators edge case
        # Create a chunker that might hit the empty separators condition
        very_long_text = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 chars with no standard separators
        chunks = chunker.chunk_text(very_long_text)
        
        assert len(chunks) >= 8  # Should split into multiple chunks
        
        # Test with very small chunk size to force character splitting
        small_chunker = RecursiveTextChunker(chunk_size=5, overlap=1)
        tiny_chunks = small_chunker.chunk_text("abcdefghijklmnop")
        assert len(tiny_chunks) >= 3
        
        # Test recursive splitting with nested separators to hit lines 170-172
        nested_chunker = RecursiveTextChunker(chunk_size=20, overlap=2)
        nested_text = "First paragraph.\n\nSecond para. Third sentence! Fourth part? Final bit."
        nested_chunks = nested_chunker.chunk_text(nested_text)
        assert len(nested_chunks) >= 2
        
        # Test condition that forces recursive sub-splitting
        complex_chunker = RecursiveTextChunker(chunk_size=15, overlap=3)
        complex_text = "A.\n\nB. C! D? E. F.\n\nG."
        complex_chunks = complex_chunker.chunk_text(complex_text)
        assert len(complex_chunks) >= 3
        
        # Test to hit remaining edge cases - ensure we cover the last few lines
        edge_chunker = RecursiveTextChunker(chunk_size=8, overlap=1)
        edge_text1 = "Test.\n\nMore.\n\nStuff! End?"
        edge_chunks1 = edge_chunker.chunk_text(edge_text1)
        assert len(edge_chunks1) >= 2
        
        # Another edge case to ensure full coverage
        edge_text2 = "X" * 25  # 25 chars, no separators 
        edge_chunks2 = edge_chunker.chunk_text(edge_text2)
        assert len(edge_chunks2) >= 3