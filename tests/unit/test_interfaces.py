"""Unit tests for interface abstract classes."""

import pytest
from abc import ABC

from enhanced_rag.interfaces.embedder import Embedder, EmbeddingResult
from enhanced_rag.interfaces.llm_client import LLMClient, Message, Role, LLMResponse
from enhanced_rag.interfaces.vector_store import VectorStore, Document, SearchResult
from enhanced_rag.interfaces.document_loader import DocumentLoader, LoadedDocument


class TestEmbedderInterface:
    """Test Embedder abstract interface."""
    
    def test_embedder_is_abstract(self):
        """Test that Embedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Embedder()
    
    def test_embedder_abstract_methods(self):
        """Test that Embedder abstract methods cannot be called directly."""
        # Verify all abstract methods raise TypeError when instantiated
        class IncompleteEmbedder(Embedder):
            """Incomplete implementation missing all methods."""
            pass
            
        with pytest.raises(TypeError):
            IncompleteEmbedder()
    
    def test_embedding_result_model(self):
        """Test EmbeddingResult data model."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            token_usage=100,
            model_name="test-embedder",
            dimensions=3
        )
        
        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.token_usage == 100
        assert result.model_name == "test-embedder"
        assert result.dimensions == 3


class TestLLMClientInterface:
    """Test LLMClient abstract interface."""
    
    def test_llm_client_is_abstract(self):
        """Test that LLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMClient()
    
    def test_llm_client_abstract_methods(self):
        """Test that LLMClient abstract methods cannot be called directly."""
        # Verify all abstract methods raise TypeError when instantiated
        class IncompleteLLMClient(LLMClient):
            """Incomplete implementation missing all methods."""
            pass
            
        with pytest.raises(TypeError):
            IncompleteLLMClient()
    
    def test_message_model(self):
        """Test Message data model."""
        message = Message(
            role=Role.USER,
            content="Hello, how are you?"
        )
        
        assert message.role == Role.USER
        assert message.content == "Hello, how are you?"
    
    def test_role_enum_values(self):
        """Test Role enum values."""
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
    
    def test_llm_response_model(self):
        """Test LLMResponse data model."""
        response = LLMResponse(
            content="Hello! I'm doing well, thank you.",
            finish_reason="stop",
            token_usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            model_name="gpt-4",
            response_time_ms=500
        )
        
        assert response.content == "Hello! I'm doing well, thank you."
        assert response.finish_reason == "stop"
        assert response.token_usage["total_tokens"] == 25
        assert response.model_name == "gpt-4"
        assert response.response_time_ms == 500


class TestVectorStoreInterface:
    """Test VectorStore abstract interface."""
    
    def test_vector_store_is_abstract(self):
        """Test that VectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorStore()
    
    def test_vector_store_abstract_methods(self):
        """Test that VectorStore abstract methods cannot be called directly."""
        # Verify all abstract methods raise TypeError when instantiated
        class IncompleteVectorStore(VectorStore):
            """Incomplete implementation missing all methods."""
            pass
            
        with pytest.raises(TypeError):
            IncompleteVectorStore()
    
    def test_document_model(self):
        """Test Document data model."""
        doc = Document(
            id="doc_123",
            content="This is a test document.",
            metadata={"source": "test.txt", "author": "tester"}
        )
        
        assert doc.id == "doc_123"
        assert doc.content == "This is a test document."
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["author"] == "tester"
    
    def test_search_result_model(self):
        """Test SearchResult data model."""
        document = Document(
            id="result_doc",
            content="Search result content",
            metadata={"relevance": "high"}
        )
        
        result = SearchResult(
            document=document,
            score=0.95,
            rank=1
        )
        
        assert result.document.id == "result_doc"
        assert result.score == 0.95
        assert result.rank == 1
        assert result.document.metadata["relevance"] == "high"


class TestDocumentLoaderInterface:
    """Test DocumentLoader abstract interface."""
    
    def test_document_loader_is_abstract(self):
        """Test that DocumentLoader cannot be instantiated directly.""" 
        with pytest.raises(TypeError):
            DocumentLoader()
    
    def test_document_loader_abstract_methods(self):
        """Test that DocumentLoader abstract methods cannot be called directly."""
        # Verify all abstract methods raise TypeError when instantiated
        class IncompleteLoader(DocumentLoader):
            """Incomplete implementation missing all methods."""
            pass
            
        with pytest.raises(TypeError):
            IncompleteLoader()
    
    def test_loaded_document_model(self):
        """Test LoadedDocument data model."""
        doc = LoadedDocument(
            content="This is the loaded document content.",
            metadata={"file_path": "/path/to/doc.txt", "file_size": 1024},
            source="/path/to/doc.txt",
            document_id="loaded_doc_456"
        )
        
        assert doc.content == "This is the loaded document content."
        assert doc.metadata["file_path"] == "/path/to/doc.txt"
        assert doc.metadata["file_size"] == 1024
        assert doc.source == "/path/to/doc.txt"
        assert doc.document_id == "loaded_doc_456"


class TestInterfaceModels:
    """Test interface model validation and edge cases."""
    
    def test_embedding_result_empty_embeddings(self):
        """Test EmbeddingResult with empty embeddings."""
        result = EmbeddingResult(
            embeddings=[],
            token_usage=0,
            model_name="test-model",
            dimensions=0
        )
        
        assert len(result.embeddings) == 0
        assert result.token_usage == 0
        assert result.dimensions == 0
    
    def test_message_with_system_role(self):
        """Test Message with system role."""
        system_message = Message(
            role=Role.SYSTEM,
            content="You are a helpful assistant."
        )
        
        assert system_message.role == Role.SYSTEM
        assert system_message.content == "You are a helpful assistant."
    
    def test_search_result_zero_score(self):
        """Test SearchResult with zero score."""
        document = Document(id="test", content="test", metadata={})
        
        result = SearchResult(
            document=document,
            score=0.0,
            rank=999
        )
        
        assert result.score == 0.0
        assert result.rank == 999
    
    def test_document_empty_metadata(self):
        """Test Document with empty metadata."""
        doc = Document(
            id="empty_meta",
            content="Content without metadata",
            metadata={}
        )
        
        assert doc.metadata == {}
        assert len(doc.metadata) == 0
    
    def test_loaded_document_empty_metadata(self):
        """Test LoadedDocument with empty metadata."""
        doc = LoadedDocument(
            content="Empty metadata content",
            metadata={},
            source="test_source",
            document_id="empty_loaded_doc"
        )
        
        assert doc.metadata == {}
        assert doc.content == "Empty metadata content"
        assert doc.source == "test_source"
    
    def test_document_with_large_content(self):
        """Test Document with large content."""
        large_content = "Large content " * 1000  
        doc = Document(
            id="large_doc",
            content=large_content,
            metadata={"size": "large"}
        )
        
        assert len(doc.content) > 10000
        assert doc.metadata["size"] == "large"
    
    def test_search_result_with_perfect_score(self):
        """Test SearchResult with perfect similarity score."""
        document = Document(id="perfect", content="Perfect match", metadata={})
        
        result = SearchResult(
            document=document,
            score=1.0,
            rank=1
        )
        
        assert result.score == 1.0
        assert result.rank == 1
    
    def test_embedding_result_with_different_dimensions(self):
        """Test EmbeddingResult with various dimensions."""
        # Test 768-dimensional embeddings (common for many models)
        result_768 = EmbeddingResult(
            embeddings=[[0.1] * 768, [0.2] * 768],
            token_usage=50,
            model_name="text-embedding-large",
            dimensions=768
        )
        
        assert result_768.dimensions == 768
        assert len(result_768.embeddings[0]) == 768
        
        # Test 1536-dimensional embeddings  
        result_1536 = EmbeddingResult(
            embeddings=[[0.1] * 1536],
            token_usage=25,
            model_name="text-embedding-3-large",
            dimensions=1536
        )
        
        assert result_1536.dimensions == 1536
        assert len(result_1536.embeddings[0]) == 1536
    
    def test_llm_response_token_usage_tracking(self):
        """Test LLMResponse token usage tracking."""
        response = LLMResponse(
            content="This is a detailed response with token tracking",
            finish_reason="stop",
            token_usage={
                "prompt_tokens": 25,
                "completion_tokens": 40,
                "total_tokens": 65,
                "cached_tokens": 5
            },
            model_name="gpt-4-turbo",
            response_time_ms=750
        )
        
        assert response.token_usage["prompt_tokens"] == 25
        assert response.token_usage["completion_tokens"] == 40
        assert response.token_usage["total_tokens"] == 65
        assert response.token_usage["cached_tokens"] == 5
        assert response.model_name == "gpt-4-turbo"
        assert response.response_time_ms == 750
    
    def test_message_role_consistency(self):
        """Test Message role consistency across different types."""
        system_msg = Message(role=Role.SYSTEM, content="You are an AI assistant.")
        user_msg = Message(role=Role.USER, content="Hello!")
        assistant_msg = Message(role=Role.ASSISTANT, content="Hi there!")
        
        assert system_msg.role.value == "system"
        assert user_msg.role.value == "user"
        assert assistant_msg.role.value == "assistant"
        
        # Test that roles are properly typed
        assert isinstance(system_msg.role, Role)
        assert isinstance(user_msg.role, Role)
        assert isinstance(assistant_msg.role, Role)


class TestAbstractInterfaceImplementation:
    """Test that all abstract interfaces properly prevent instantiation."""
    
    def test_all_interfaces_are_abstract(self):
        """Comprehensive test that all abstract interfaces raise TypeError."""
        # Test all abstract interfaces at once to ensure all pass statements are covered
        abstract_classes = [
            Embedder,
            LLMClient, 
            VectorStore,
            DocumentLoader
        ]
        
        for abstract_class in abstract_classes:
            with pytest.raises(TypeError, match="Can't instantiate abstract class"):
                abstract_class()
        
        # Also test that incomplete implementations fail
        class IncompleteEmbedder(Embedder):
            pass
            
        class IncompleteLLMClient(LLMClient):
            pass
            
        class IncompleteVectorStore(VectorStore):
            pass
            
        class IncompleteDocumentLoader(DocumentLoader):
            pass
        
        incomplete_classes = [
            IncompleteEmbedder,
            IncompleteLLMClient,
            IncompleteVectorStore,
            IncompleteDocumentLoader
        ]
        
        for incomplete_class in incomplete_classes:
            with pytest.raises(TypeError):
                incomplete_class()
    
    def test_interface_data_models_comprehensive(self):
        """Comprehensive test of all interface data models."""
        # Test EmbeddingResult with edge cases
        embedding_result = EmbeddingResult(
            embeddings=[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]],
            token_usage=150,
            model_name="comprehensive-test-model",
            dimensions=3
        )
        
        assert len(embedding_result.embeddings) == 2
        assert embedding_result.embeddings[1][0] == -0.4  # Test negative values
        assert embedding_result.dimensions == 3
        
        # Test Message with edge cases
        long_content = "This is a very long message content " * 50
        long_message = Message(
            role=Role.USER,
            content=long_content
        )
        
        assert len(long_message.content) > 1000
        assert long_message.role == Role.USER
        
        # Test SearchResult with edge cases
        search_doc = Document(
            id="comprehensive-search-doc",
            content="Comprehensive search result content with special chars: áéíóú",
            metadata={"special": True, "priority": 9999}
        )
        
        search_result = SearchResult(
            document=search_doc,
            score=0.999999,
            rank=1
        )
        
        assert search_result.score > 0.999
        assert search_result.document.metadata["priority"] == 9999
        
        # Test LoadedDocument with comprehensive metadata
        loaded_doc = LoadedDocument(
            content="Comprehensive loaded document content",
            metadata={
                "file_extension": ".comprehensive",
                "processing_timestamp": 1640995200,
                "content_length": 35,
                "encoding_detected": "utf-8",
                "processing_successful": True
            },
            source="/comprehensive/test/path.comprehensive",
            document_id="comprehensive-loaded-doc-12345"
        )
        
        assert loaded_doc.metadata["processing_successful"] is True
        assert loaded_doc.document_id.endswith("12345")
        assert loaded_doc.source.endswith(".comprehensive")


class TestInterfaceEdgeCases:
    """Test edge cases to improve interface coverage."""
    
    def test_role_enum_string_representation(self):
        """Test Role enum string representations."""
        # Test role values (not string representation)
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user" 
        assert Role.ASSISTANT.value == "assistant"
        
        # Test role equality
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
    
    def test_embedding_result_edge_cases(self):
        """Test EmbeddingResult with various edge cases."""
        # Test with very high dimensions
        high_dim_result = EmbeddingResult(
            embeddings=[[0.1] * 4096],
            token_usage=1000,
            model_name="high-dimensional-model",
            dimensions=4096
        )
        
        assert high_dim_result.dimensions == 4096
        assert len(high_dim_result.embeddings[0]) == 4096
        
        # Test with multiple embeddings of different patterns
        pattern_result = EmbeddingResult(
            embeddings=[
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0], 
                [-1.0, 0.0, 1.0]
            ],
            token_usage=75,
            model_name="pattern-test-model",
            dimensions=3
        )
        
        assert len(pattern_result.embeddings) == 3
        assert pattern_result.embeddings[0][0] == 1.0
        assert pattern_result.embeddings[1][1] == 1.0
        assert pattern_result.embeddings[2][2] == 1.0