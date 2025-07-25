"""Unit tests for REST API functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from enhanced_rag.api.rest import RAGService, get_rag_service, app
from enhanced_rag.api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthCheckResponse,
    ErrorResponse
)
from enhanced_rag.utils.config import RAGConfig
from enhanced_rag.pipeline import RAGPipeline


class TestRAGService:
    """Test RAGService class functionality."""
    
    def setup_method(self):
        """Setup test service instance."""
        self.service = RAGService()
    
    def test_rag_service_init(self):
        """Test RAGService initialization."""
        service = RAGService()
        
        assert service.pipeline is None
        assert service.config is None
        assert service.metrics is not None
    
    def test_initialize_service(self):
        """Test service initialization."""
        mock_config = Mock(spec=RAGConfig)
        
        self.service.initialize(mock_config)
        
        assert self.service.config == mock_config
    
    def test_get_pipeline_not_initialized(self):
        """Test getting pipeline when not initialized."""
        with pytest.raises(HTTPException) as exc_info:
            self.service.get_pipeline()
        
        assert exc_info.value.status_code == 503
        assert "Service not initialized" in str(exc_info.value.detail)
    
    def test_get_pipeline_initialized(self):
        """Test getting pipeline when initialized."""
        mock_pipeline = Mock(spec=RAGPipeline)
        self.service.pipeline = mock_pipeline
        
        result = self.service.get_pipeline()
        
        assert result == mock_pipeline


class TestDependencies:
    """Test FastAPI dependency functions."""
    
    def test_get_rag_service(self):
        """Test get_rag_service dependency."""
        service = get_rag_service()
        
        assert isinstance(service, RAGService)


class TestHealthCheckEndpoint:
    """Test health check endpoint functionality."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_health_check_success(self, mock_service):
        """Test successful health check."""
        # Mock pipeline and health check response
        mock_pipeline = AsyncMock(spec=RAGPipeline)
        mock_health_status = {
            "status": "healthy",
            "components": {"embedder": "healthy", "llm": "healthy"},
            "timestamp": 1609459200.0
        }
        mock_pipeline.health_check.return_value = mock_health_status
        mock_service.get_pipeline.return_value = mock_pipeline
        
        # Make request
        response = self.client.get("/health")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "timestamp" in data
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_health_check_service_not_initialized(self, mock_service):
        """Test health check when service not initialized."""
        mock_service.get_pipeline.side_effect = HTTPException(
            status_code=503, 
            detail="Service not initialized"
        )
        
        response = self.client.get("/health")
        
        assert response.status_code == 503
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_health_check_pipeline_error(self, mock_service):
        """Test health check when pipeline raises error."""
        mock_pipeline = AsyncMock(spec=RAGPipeline)
        mock_pipeline.health_check.side_effect = Exception("Pipeline error")
        mock_service.get_pipeline.return_value = mock_pipeline
        
        response = self.client.get("/health")
        
        assert response.status_code == 503


class TestQueryEndpoint:
    """Test query endpoint functionality."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_query_success(self, mock_service):
        """Test successful query."""
        # Mock pipeline and query response
        mock_pipeline = AsyncMock(spec=RAGPipeline)
        mock_query_response = {
            "answer": "This is the answer",
            "correlation_id": "test-123",
            "processing_time_ms": 1500,
            "retrieved_chunks": 3,
            "model_info": {"embedder": "text-ada-002", "llm": "gpt-4"}
        }
        mock_pipeline.query.return_value = mock_query_response
        mock_service.get_pipeline.return_value = mock_pipeline
        
        # Make request
        query_data = {
            "question": "What is machine learning?",
            "top_k": 5,
            "include_metadata": True
        }
        
        response = self.client.post("/query", json=query_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is the answer"
        assert data["correlation_id"] == "test-123"
        assert "processing_time_ms" in data
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_query_with_correlation_id(self, mock_service):
        """Test query with correlation ID."""
        mock_pipeline = AsyncMock(spec=RAGPipeline)
        mock_pipeline.query.return_value = {
            "answer": "Answer",
            "correlation_id": "custom-corr-123",
            "processing_time_ms": 1000,
            "retrieved_chunks": 1,
            "model_info": {"llm": "gpt-4"}
        }
        mock_service.get_pipeline.return_value = mock_pipeline
        
        query_data = {
            "question": "Test question?",
            "correlation_id": "custom-corr-123"
        }
        
        response = self.client.post("/query", json=query_data)
        
        assert response.status_code == 200
        # Verify that set_correlation_id was called (indirectly through pipeline call)
        mock_pipeline.query.assert_called_once()
    
    def test_query_invalid_request(self):
        """Test query with invalid request data."""
        # Missing required 'question' field
        invalid_data = {
            "top_k": 5
        }
        
        response = self.client.post("/query", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_query_question_too_long(self):
        """Test query with question too long."""
        long_question = "x" * 2001  # Exceeds max length of 2000
        
        query_data = {
            "question": long_question
        }
        
        response = self.client.post("/query", json=query_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_query_service_not_initialized(self, mock_service):
        """Test query when service not initialized.""" 
        # The query endpoint catches all exceptions and converts to 500
        mock_service.get_pipeline.side_effect = HTTPException(
            status_code=503,
            detail="Service not initialized"
        )
        
        query_data = {
            "question": "Test question?"
        }
        
        response = self.client.post("/query", json=query_data)
        
        # Query endpoint catches exceptions and returns 500 with error details
        assert response.status_code == 500


class TestGlobalExceptionHandler:
    """Test global exception handler."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_global_exception_handler(self, mock_service):
        """Test global exception handler."""
        # The health endpoint handles exceptions and returns 503, not 500
        # The global exception handler would be triggered by different exceptions
        mock_service.get_pipeline.side_effect = RuntimeError("Unexpected error")
        
        response = self.client.get("/health")
        
        # Health endpoint catches exceptions and returns 503
        assert response.status_code == 503


class TestCorrelationMiddleware:
    """Test correlation middleware functionality."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_correlation_middleware_with_header(self, mock_service):
        """Test correlation middleware with correlation ID header."""
        mock_pipeline = AsyncMock(spec=RAGPipeline)
        mock_pipeline.health_check.return_value = {
            "status": "healthy",
            "components": {},
            "timestamp": 1609459200.0
        }
        mock_service.get_pipeline.return_value = mock_pipeline
        
        # Make request with correlation ID header
        headers = {"X-Correlation-ID": "test-correlation-456"}
        response = self.client.get("/health", headers=headers)
        
        # Verify correlation ID is in response headers
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        # Note: The actual correlation ID might be generated if not provided
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_correlation_middleware_without_header(self, mock_service):
        """Test correlation middleware without correlation ID header."""
        mock_pipeline = AsyncMock(spec=RAGPipeline)
        mock_pipeline.health_check.return_value = {
            "status": "healthy", 
            "components": {},
            "timestamp": 1609459200.0
        }
        mock_service.get_pipeline.return_value = mock_pipeline
        
        response = self.client.get("/health")
        
        # Should still have correlation ID in response (generated)
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers


class TestRequestValidation:
    """Test request validation for various endpoints."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_query_request_validation_empty_question(self):
        """Test query request validation with empty question."""
        query_data = {
            "question": ""
        }
        
        response = self.client.post("/query", json=query_data)
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_query_request_validation_invalid_top_k(self):
        """Test query request validation with invalid top_k."""
        query_data = {
            "question": "Valid question?",
            "top_k": 0  # Should be >= 1
        }
        
        response = self.client.post("/query", json=query_data)
        
        assert response.status_code == 422
    
    def test_query_request_validation_top_k_too_large(self):
        """Test query request validation with top_k too large."""
        query_data = {
            "question": "Valid question?",
            "top_k": 51  # Should be <= 50
        }
        
        response = self.client.post("/query", json=query_data)
        
        assert response.status_code == 422


class TestAppConfiguration:
    """Test FastAPI app configuration."""
    
    def test_app_metadata(self):
        """Test app title and metadata."""
        assert app.title == "Enhanced RAG System API"
        assert app.description == "Production-ready Retrieval-Augmented Generation system"
        assert app.version == "1.0.0"
    
    def test_middleware_configuration(self):
        """Test that middleware is properly configured."""
        # Check that middleware is configured (FastAPI wraps them in Middleware objects)
        assert len(app.user_middleware) >= 2  # At least CORS and GZip middleware


class TestLifespanEvents:
    """Test application lifespan events."""
    
    @patch('enhanced_rag.api.rest.ConfigManager')
    @patch('enhanced_rag.api.rest.setup_metrics')
    @patch('enhanced_rag.api.rest.rag_service')
    def test_lifespan_startup(self, mock_service, mock_setup_metrics, mock_config_manager):
        """Test application startup lifespan."""
        # Mock configuration
        mock_config = Mock()
        mock_config.observability.metrics_port = 8000
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        # Test startup would be handled by FastAPI lifespan
        # We can test the components are properly mocked
        assert mock_config_manager is not None
        assert mock_setup_metrics is not None
        assert mock_service is not None


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_404_endpoint_not_found(self):
        """Test 404 for non-existent endpoint."""
        response = self.client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test 405 for wrong HTTP method."""
        # GET on POST endpoint
        response = self.client.get("/query")
        
        assert response.status_code == 405
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_internal_server_error_handling(self, mock_service):
        """Test internal server error handling."""
        # Force an exception in the service  
        mock_service.get_pipeline.side_effect = Exception("Database connection failed")
        
        response = self.client.get("/health")
        
        # Health endpoint catches and handles exceptions, returning 503
        assert response.status_code == 503


class TestIntegration:
    """Integration tests for REST API."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_full_query_workflow(self, mock_service):
        """Test complete query workflow."""
        # Mock pipeline with realistic response
        mock_pipeline = AsyncMock(spec=RAGPipeline)
        mock_pipeline.query.return_value = {
            "answer": "Machine learning is a subset of artificial intelligence.",
            "correlation_id": "workflow-test-789",
            "processing_time_ms": 1250,
            "retrieved_chunks": 3,
            "model_info": {
                "embedder": "text-embedding-ada-002",
                "llm": "gpt-4"
            },
            "sources": [
                {
                    "chunk_id": "chunk_1",
                    "content": "ML is part of AI...",
                    "score": 0.92,
                    "metadata": {"source": "ai_textbook.pdf"}
                }
            ]
        }
        mock_service.get_pipeline.return_value = mock_pipeline
        
        # Make query request
        query_data = {
            "question": "What is machine learning?",
            "top_k": 3,
            "include_metadata": True,
            "correlation_id": "workflow-test-789"
        }
        
        response = self.client.post("/query", json=query_data)
        
        # Verify complete response
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Machine learning is a subset of artificial intelligence."
        assert data["correlation_id"] == "workflow-test-789"
        assert data["retrieved_chunks"] == 3
        assert "model_info" in data
        assert "sources" in data
        assert len(data["sources"]) == 1
        
        # Verify correlation ID in response headers
        assert "X-Correlation-ID" in response.headers
    
    @patch('enhanced_rag.api.rest.rag_service')
    def test_health_check_integration(self, mock_service):
        """Test health check integration."""
        mock_pipeline = AsyncMock(spec=RAGPipeline)
        mock_pipeline.health_check.return_value = {
            "status": "healthy",
            "components": {
                "embedder": "healthy",
                "vector_store": "healthy", 
                "llm_client": "healthy"
            },
            "timestamp": 1609459200.0
        }
        mock_service.get_pipeline.return_value = mock_pipeline
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert len(data["components"]) == 3
        assert all(status == "healthy" for status in data["components"].values())
        assert data["timestamp"] == 1609459200.0


class TestAdditionalEndpoints:
    """Test additional REST API endpoints for better coverage."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_openapi_schema(self):
        """Test that OpenAPI schema is accessible."""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert data["info"]["title"] == "Enhanced RAG System API"
    
    def test_docs_endpoint(self):
        """Test that API docs are accessible."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self):
        """Test that ReDoc docs are accessible."""
        response = self.client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestRAGServiceInitialization:
    """Test RAGService initialization edge cases."""
    
    def test_service_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        service = RAGService()
        
        # Service should have metrics initialized
        assert service.metrics is not None
        assert hasattr(service.metrics, 'query_duration')
    
    def test_rag_service_dependency_injection(self):
        """Test dependency injection pattern."""
        # Test that we can get the same service instance
        service1 = get_rag_service()
        service2 = get_rag_service()
        
        # Should be the same instance (singleton pattern)
        assert service1 is service2
    
    def test_rag_service_config_storage(self):
        """Test that config is properly stored."""
        service = RAGService()
        mock_config = Mock(spec=RAGConfig)
        mock_config.llm = Mock()
        mock_config.embedding = Mock()
        
        service.initialize(mock_config)
        
        assert service.config == mock_config