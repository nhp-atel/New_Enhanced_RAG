"""Unit tests for RAG pipeline functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import asyncio

from enhanced_rag.pipeline import RAGPipeline
from enhanced_rag.interfaces import Embedder, VectorStore, LLMClient, DocumentLoader
from enhanced_rag.interfaces.vector_store import Document, SearchResult
from enhanced_rag.interfaces.llm_client import Message, Role, LLMResponse
from enhanced_rag.interfaces.embedder import EmbeddingResult
from enhanced_rag.interfaces.document_loader import LoadedDocument
from enhanced_rag.utils.config import RAGConfig, ChunkingConfig, EmbeddingConfig, VectorStoreConfig, LLMConfig, RetrievalConfig
from enhanced_rag.core.chunking import TextChunk


@pytest.fixture
def mock_config():
    """Mock RAG configuration."""
    config = Mock(spec=RAGConfig)
    config.chunking = Mock(spec=ChunkingConfig)
    config.chunking.strategy = "recursive"
    config.chunking.chunk_size = 1000
    config.chunking.overlap = 200
    
    config.embedding = Mock(spec=EmbeddingConfig)
    config.embedding.batch_size = 100
    
    config.vector_store = Mock(spec=VectorStoreConfig)
    config.vector_store.persistence_path = "/tmp/test_index"
    
    config.llm = Mock(spec=LLMConfig)
    config.llm.max_tokens = 2048
    config.llm.temperature = 0.1
    
    config.retrieval = Mock(spec=RetrievalConfig)
    config.retrieval.top_k = 5
    config.retrieval.score_threshold = 0.7
    
    return config


@pytest.fixture
def mock_embedder():
    """Mock embedder implementation."""
    embedder = Mock(spec=Embedder)
    embedder.model_name = "test-embedder"
    embedder.embedding_dimension = 384
    
    # Mock embed_texts method
    async def mock_embed_texts(texts, batch_size=None):
        return EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3] * 128 for _ in texts],  # 384-dim embeddings
            token_usage=len(texts) * 10,
            model_name="test-embedder",
            dimensions=384
        )
    
    embedder.embed_texts = AsyncMock(side_effect=mock_embed_texts)
    
    # Mock embed_query method
    async def mock_embed_query(query):
        return [0.1, 0.2, 0.3] * 128  # 384-dim embedding
    
    embedder.embed_query = AsyncMock(side_effect=mock_embed_query)
    
    return embedder


@pytest.fixture
def mock_vector_store():
    """Mock vector store implementation."""
    vector_store = Mock(spec=VectorStore)
    
    # Mock storage
    stored_documents = []
    
    async def mock_add_documents(documents, embeddings):
        stored_documents.extend(documents)
    
    async def mock_similarity_search(query_embedding, top_k=10, filters=None):
        # Return mock search results
        results = []
        for i, doc in enumerate(stored_documents[:top_k]):
            result = SearchResult(
                document=doc,
                score=0.9 - (i * 0.1),  # Decreasing scores
                rank=i
            )
            results.append(result)
        return results
    
    async def mock_save_index(path):
        pass
    
    def mock_get_stats():
        return {"document_count": len(stored_documents), "index_size": "1MB"}
    
    vector_store.add_documents = AsyncMock(side_effect=mock_add_documents)
    vector_store.similarity_search = AsyncMock(side_effect=mock_similarity_search)
    vector_store.save_index = AsyncMock(side_effect=mock_save_index)
    vector_store.get_stats = Mock(side_effect=mock_get_stats)
    
    return vector_store


@pytest.fixture
def mock_llm_client():
    """Mock LLM client implementation."""
    llm_client = Mock(spec=LLMClient)
    llm_client.model_name = "test-llm"
    llm_client.max_context_length = 8000
    
    async def mock_generate_response(messages, max_tokens=None, temperature=0.7, **kwargs):
        # Simple mock response based on the last message
        last_message = messages[-1].content if messages else "No context provided"
        
        return LLMResponse(
            content=f"Based on the provided context, here is the answer to your question.",
            finish_reason="stop",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model_name="test-llm",
            response_time_ms=500
        )
    
    llm_client.generate_response = AsyncMock(side_effect=mock_generate_response)
    
    return llm_client


@pytest.fixture
def mock_document_loader():
    """Mock document loader implementation."""
    loader = Mock(spec=DocumentLoader)
    
    async def mock_load_document(source):
        return LoadedDocument(
            content=f"Sample content from {source}",
            metadata={"file_path": source, "file_size": 1000},
            source=source,
            document_id=f"doc_{hash(source)}"
        )
    
    loader.load_document = AsyncMock(side_effect=mock_load_document)
    loader.supports_source = Mock(return_value=True)
    loader.supported_extensions = [".txt", ".pdf"]
    
    return loader


@pytest.fixture
def rag_pipeline(mock_config, mock_embedder, mock_vector_store, mock_llm_client, mock_document_loader):
    """RAG pipeline with mocked dependencies."""
    document_loaders = {".txt": mock_document_loader, ".pdf": mock_document_loader}
    
    pipeline = RAGPipeline(
        config=mock_config,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        document_loaders=document_loaders
    )
    
    return pipeline


class TestRAGPipeline:
    """Test cases for RAG pipeline."""
    
    def test_pipeline_initialization(self, rag_pipeline, mock_config):
        """Test pipeline initialization."""
        assert rag_pipeline.config == mock_config
        assert rag_pipeline.embedder is not None
        assert rag_pipeline.vector_store is not None
        assert rag_pipeline.llm_client is not None
        assert len(rag_pipeline.document_loaders) == 2
    
    @pytest.mark.asyncio
    async def test_ingest_documents_success(self, rag_pipeline):
        """Test successful document ingestion."""
        document_paths = ["test1.txt", "test2.txt"]
        
        with patch.object(rag_pipeline, '_load_document') as mock_load:
            # Mock loaded documents
            mock_load.side_effect = [
                LoadedDocument(
                    content="Content 1",
                    metadata={"source": "test1.txt"},
                    source="test1.txt",
                    document_id="doc1"
                ),
                LoadedDocument(
                    content="Content 2",
                    metadata={"source": "test2.txt"},
                    source="test2.txt",
                    document_id="doc2"
                )
            ]
            
            stats = await rag_pipeline.ingest_documents(document_paths, batch_size=2)
        
        assert stats["documents_processed"] == 2
        assert stats["chunks_created"] >= 2  # At least 1 chunk per document
        assert stats["embeddings_generated"] >= 2
        assert "processing_time_seconds" in stats
        assert len(stats["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_ingest_documents_with_errors(self, rag_pipeline):
        """Test document ingestion with some errors."""
        document_paths = ["valid.txt", "invalid.txt"]
        
        with patch.object(rag_pipeline, '_load_document') as mock_load:
            def side_effect(path):
                if path == "invalid.txt":
                    raise ValueError("Invalid document")
                return LoadedDocument(
                    content="Valid content",
                    metadata={"source": path},
                    source=path,
                    document_id="doc1"
                )
            
            mock_load.side_effect = side_effect
            
            stats = await rag_pipeline.ingest_documents(document_paths, batch_size=2)
        
        assert stats["documents_processed"] == 1  # Only valid document processed
        assert len(stats["errors"]) == 1
        assert "invalid.txt" in stats["errors"][0]
    
    @pytest.mark.asyncio
    async def test_query_success(self, rag_pipeline):
        """Test successful query processing."""
        # First ingest some documents
        await self._ingest_test_documents(rag_pipeline)
        
        question = "What is the test content about?"
        
        response = await rag_pipeline.query(question)
        
        assert "answer" in response
        assert "correlation_id" in response
        assert "processing_time_ms" in response
        assert "retrieved_chunks" in response
        assert response["retrieved_chunks"] >= 0
        assert "model_info" in response
    
    @pytest.mark.asyncio
    async def test_query_no_relevant_context(self, rag_pipeline):
        """Test query when no relevant context is found."""
        # Mock vector store to return no results
        rag_pipeline.vector_store.similarity_search = AsyncMock(return_value=[])
        
        question = "What is the test content about?"
        
        response = await rag_pipeline.query(question)
        
        assert "no_relevant_context" in response
        assert response["no_relevant_context"] is True
        assert response["retrieved_chunks"] == 0
    
    @pytest.mark.asyncio
    async def test_query_with_filters(self, rag_pipeline):
        """Test query with metadata filters."""
        await self._ingest_test_documents(rag_pipeline)
        
        question = "Test question"
        filters = {"source": "test.txt"}
        
        response = await rag_pipeline.query(question, filters=filters)
        
        # Verify filters were passed to vector store
        rag_pipeline.vector_store.similarity_search.assert_called()
        call_args = rag_pipeline.vector_store.similarity_search.call_args
        assert call_args[1]["filters"] == filters
    
    @pytest.mark.asyncio
    async def test_query_with_custom_top_k(self, rag_pipeline):
        """Test query with custom top_k parameter."""
        await self._ingest_test_documents(rag_pipeline)
        
        question = "Test question"
        top_k = 3
        
        response = await rag_pipeline.query(question, top_k=top_k)
        
        # Verify top_k was passed to vector store
        call_args = rag_pipeline.vector_store.similarity_search.call_args
        assert call_args[1]["top_k"] == top_k
    
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, rag_pipeline):
        """Test health check when all components are healthy."""
        health = await rag_pipeline.health_check()
        
        assert health["status"] == "healthy"
        assert "components" in health
        assert "timestamp" in health
        
        # All components should be healthy
        for component_status in health["components"].values():
            assert component_status == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_with_failures(self, rag_pipeline):
        """Test health check when some components fail."""
        # Make embedder fail
        rag_pipeline.embedder.embed_query = AsyncMock(side_effect=Exception("Embedder error"))
        
        health = await rag_pipeline.health_check()
        
        assert health["status"] == "degraded"
        assert "unhealthy: Embedder error" in health["components"]["embedder"]
    
    @pytest.mark.asyncio
    async def test_get_stats(self, rag_pipeline):
        """Test getting pipeline statistics."""
        stats = await rag_pipeline.get_stats()
        
        assert "pipeline_config" in stats
        assert "vector_store" in stats
        assert "embedding_dimension" in stats
        
        # Check pipeline config
        config = stats["pipeline_config"]
        assert "chunk_size" in config
        assert "embedding_model" in config
        assert "llm_model" in config
    
    @pytest.mark.asyncio
    async def test_embedder_retry_logic(self, rag_pipeline):
        """Test retry logic for embedder failures."""
        # Make embedder fail first two times, succeed on third
        call_count = 0
        
        async def failing_embed_query(query):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")  # Use ConnectionError which is retryable
            return [0.1] * 384
        
        rag_pipeline.embedder.embed_query = AsyncMock(side_effect=failing_embed_query)
        
        # This should succeed after retries
        embedding = await rag_pipeline._generate_query_embedding_with_retry("test")
        
        assert len(embedding) == 384
        assert call_count == 3  # Two failures + one success
    
    @pytest.mark.asyncio
    async def test_llm_retry_logic(self, rag_pipeline):
        """Test retry logic for LLM failures."""
        call_count = 0
        
        async def failing_generate_response(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("API error")  # Use ConnectionError which is retryable
            return LLMResponse(
                content="Success after retry",
                finish_reason="stop",
                model_name="test-llm",
                response_time_ms=100
            )
        
        rag_pipeline.llm_client.generate_response = AsyncMock(side_effect=failing_generate_response)
        
        # Create mock search results
        mock_results = [
            SearchResult(
                document=Document(id="test", content="test content", metadata={}),
                score=0.9,
                rank=0
            )
        ]
        
        response = await rag_pipeline._generate_answer_with_retry("test question", mock_results)
        
        assert response.content == "Success after retry"
        assert call_count == 3
    
    async def _ingest_test_documents(self, pipeline):
        """Helper method to ingest test documents."""
        # Mock some documents in the vector store
        test_docs = [
            Document(
                id="chunk_1",
                content="This is test content about machine learning.",
                metadata={"source": "test1.txt", "chunk_index": 0}
            ),
            Document(
                id="chunk_2", 
                content="This is more test content about natural language processing.",
                metadata={"source": "test2.txt", "chunk_index": 0}
            )
        ]
        
        # Add to mock vector store
        await pipeline.vector_store.add_documents(test_docs, [[0.1] * 384] * len(test_docs))


class TestRAGPipelineError:
    """Test error handling in RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_query_with_embedder_failure(self, rag_pipeline):
        """Test query handling when embedder fails."""
        rag_pipeline.embedder.embed_query = AsyncMock(side_effect=Exception("Embedder failed"))
        
        response = await rag_pipeline.query("test question")
        
        assert "error" in response
        assert "Embedder failed" in response["error"]
    
    @pytest.mark.asyncio
    async def test_query_with_vector_store_failure(self, rag_pipeline):
        """Test query handling when vector store fails."""
        rag_pipeline.vector_store.similarity_search = AsyncMock(side_effect=Exception("Vector store failed"))
        
        response = await rag_pipeline.query("test question")
        
        assert "error" in response
        assert "Vector store failed" in response["error"]
    
    @pytest.mark.asyncio
    async def test_query_with_llm_failure(self, rag_pipeline):
        """Test query handling when LLM fails."""
        # Mock successful retrieval
        await self._setup_successful_retrieval(rag_pipeline)
        
        # But make LLM fail
        rag_pipeline.llm_client.generate_response = AsyncMock(side_effect=Exception("LLM failed"))
        
        response = await rag_pipeline.query("test question")
        
        assert "error" in response
        assert "LLM failed" in response["error"]
    
    async def _setup_successful_retrieval(self, pipeline):
        """Helper to setup successful retrieval."""
        mock_results = [
            SearchResult(
                document=Document(id="test", content="test", metadata={}),
                score=0.9,
                rank=0
            )
        ]
        pipeline.vector_store.similarity_search = AsyncMock(return_value=mock_results)


class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, rag_pipeline):
        """Test complete end-to-end pipeline flow."""
        
        # Step 1: Ingest documents
        document_paths = ["test_doc.txt"]
        
        with patch.object(rag_pipeline, '_load_document') as mock_load:
            mock_load.return_value = LoadedDocument(
                content="This is a comprehensive test document about artificial intelligence and machine learning.",
                metadata={"source": "test_doc.txt"},
                source="test_doc.txt",
                document_id="test_doc"
            )
            
            ingestion_stats = await rag_pipeline.ingest_documents(document_paths)
        
        # Verify ingestion
        assert ingestion_stats["documents_processed"] == 1
        assert ingestion_stats["chunks_created"] >= 1
        
        # Step 2: Query the system
        response = await rag_pipeline.query("What is this document about?")
        
        # Verify query response
        assert "answer" in response
        assert response["retrieved_chunks"] > 0
        assert "correlation_id" in response
        
        # Step 3: Check system health
        health = await rag_pipeline.health_check()
        assert health["status"] in ["healthy", "degraded"]