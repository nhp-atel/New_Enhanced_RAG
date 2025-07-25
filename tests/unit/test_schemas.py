"""Unit tests for API schemas."""

import pytest
from pydantic import ValidationError
from typing import Dict, Any

from enhanced_rag.api.schemas import (
    QueryRequest,
    SourceInfo,
    QueryResponse,
    ErrorResponse,
    IngestionRequest,
    IngestionResponse,
    HealthCheckResponse,
    StatsResponse,
    ConfigUpdateRequest,
    ConfigResponse,
    StreamingQueryResponse,
    DocumentValidationRequest,
    DocumentValidationResponse,
    BatchQueryRequest,
    BatchQueryResponse,
    SearchFilter,
    AdvancedQueryRequest,
    SystemCommand,
    AdminRequest,
    AdminResponse,
    MetricsResponse,
    LogsRequest,
    LogEntry,
    LogsResponse
)


class TestQueryRequest:
    """Test QueryRequest schema validation."""
    
    def test_query_request_valid(self):
        """Test valid query request."""
        request = QueryRequest(
            question="What is machine learning?",
            top_k=5,
            filters={"source": "textbook.pdf"},
            include_metadata=True,
            correlation_id="test-123"
        )
        
        assert request.question == "What is machine learning?"
        assert request.top_k == 5
        assert request.filters == {"source": "textbook.pdf"}
        assert request.include_metadata is True
        assert request.correlation_id == "test-123"
    
    def test_query_request_minimal(self):
        """Test minimal valid query request."""
        request = QueryRequest(question="Test question")
        
        assert request.question == "Test question"
        assert request.top_k is None
        assert request.filters is None
        assert request.include_metadata is True
        assert request.correlation_id is None
    
    def test_query_request_empty_question(self):
        """Test query request with empty question."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="")
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("question",)
        assert "at least 1 character" in errors[0]["msg"]
    
    def test_query_request_question_too_long(self):
        """Test query request with question too long."""
        long_question = "x" * 2001
        
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question=long_question)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("question",)
        assert "at most 2000 characters" in errors[0]["msg"]
    
    def test_query_request_invalid_top_k(self):
        """Test query request with invalid top_k."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="Test", top_k=0)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("top_k",)
        assert "greater than or equal to 1" in errors[0]["msg"]
        
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="Test", top_k=51)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("top_k",)
        assert "less than or equal to 50" in errors[0]["msg"]


class TestSourceInfo:
    """Test SourceInfo schema validation."""
    
    def test_source_info_valid(self):
        """Test valid source info."""
        source = SourceInfo(
            chunk_id="chunk_123",
            content="This is the content of the chunk.",
            score=0.85,
            metadata={"source": "doc.pdf", "page": 1}
        )
        
        assert source.chunk_id == "chunk_123"
        assert source.content == "This is the content of the chunk."
        assert source.score == 0.85
        assert source.metadata == {"source": "doc.pdf", "page": 1}
    
    def test_source_info_invalid_score(self):
        """Test source info with invalid score."""
        with pytest.raises(ValidationError) as exc_info:
            SourceInfo(
                chunk_id="test",
                content="test content",
                score=-0.1,
                metadata={}
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("score",)
        assert "greater than or equal to 0" in errors[0]["msg"]
        
        with pytest.raises(ValidationError) as exc_info:
            SourceInfo(
                chunk_id="test",
                content="test content", 
                score=1.1,
                metadata={}
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("score",)
        assert "less than or equal to 1" in errors[0]["msg"]


class TestQueryResponse:
    """Test QueryResponse schema validation."""
    
    def test_query_response_valid(self):
        """Test valid query response."""
        sources = [
            SourceInfo(
                chunk_id="chunk_1",
                content="Test content",
                score=0.9,
                metadata={"source": "test.pdf"}
            )
        ]
        
        response = QueryResponse(
            answer="This is the answer",
            correlation_id="corr-123",
            processing_time_ms=1500,
            retrieved_chunks=1,
            model_info={"embedder": "text-ada-002", "llm": "gpt-4"},
            sources=sources,
            no_relevant_context=False
        )
        
        assert response.answer == "This is the answer"
        assert response.correlation_id == "corr-123"
        assert response.processing_time_ms == 1500
        assert response.retrieved_chunks == 1
        assert response.model_info == {"embedder": "text-ada-002", "llm": "gpt-4"}
        assert len(response.sources) == 1
        assert response.no_relevant_context is False
    
    def test_query_response_minimal(self):
        """Test minimal query response."""
        response = QueryResponse(
            answer="Answer",
            correlation_id="corr-456",
            processing_time_ms=100,
            retrieved_chunks=0,
            model_info={"llm": "gpt-3.5"}
        )
        
        assert response.sources is None
        assert response.no_relevant_context is None


class TestErrorResponse:
    """Test ErrorResponse schema validation."""
    
    def test_error_response_valid(self):
        """Test valid error response."""
        response = ErrorResponse(
            error="Something went wrong",
            error_type="ValidationError",
            correlation_id="err-123",
            details={"field": "question", "issue": "too short"}
        )
        
        assert response.error == "Something went wrong"
        assert response.error_type == "ValidationError"
        assert response.correlation_id == "err-123"
        assert response.details == {"field": "question", "issue": "too short"}
    
    def test_error_response_minimal(self):
        """Test minimal error response."""
        response = ErrorResponse(
            error="Error occurred",
            error_type="GenericError"
        )
        
        assert response.correlation_id is None
        assert response.details is None


class TestIngestionRequest:
    """Test IngestionRequest schema validation."""
    
    def test_ingestion_request_valid(self):
        """Test valid ingestion request."""
        request = IngestionRequest(
            document_paths=["doc1.pdf", "doc2.txt"],
            batch_size=20,
            overwrite_existing=True
        )
        
        assert request.document_paths == ["doc1.pdf", "doc2.txt"]
        assert request.batch_size == 20
        assert request.overwrite_existing is True
    
    def test_ingestion_request_minimal(self):
        """Test minimal ingestion request."""
        request = IngestionRequest(
            document_paths=["single_doc.pdf"]
        )
        
        assert request.batch_size == 10  # default
        assert request.overwrite_existing is False  # default
    
    def test_ingestion_request_empty_paths(self):
        """Test ingestion request with empty document paths."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionRequest(document_paths=[])
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("document_paths",)
        assert "at least 1 item" in errors[0]["msg"]
    
    def test_ingestion_request_invalid_batch_size(self):
        """Test ingestion request with invalid batch size."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionRequest(
                document_paths=["doc.pdf"],
                batch_size=0
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("batch_size",)
        assert "greater than or equal to 1" in errors[0]["msg"]
        
        with pytest.raises(ValidationError) as exc_info:
            IngestionRequest(
                document_paths=["doc.pdf"],
                batch_size=101
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("batch_size",)
        assert "less than or equal to 100" in errors[0]["msg"]


class TestIngestionResponse:
    """Test IngestionResponse schema validation."""
    
    def test_ingestion_response_valid(self):
        """Test valid ingestion response."""
        response = IngestionResponse(
            documents_processed=5,
            chunks_created=50,
            embeddings_generated=50,
            processing_time_s=12.5,
            errors=["Failed to process doc3.pdf"],
            correlation_id="ing-789"
        )
        
        assert response.documents_processed == 5
        assert response.chunks_created == 50
        assert response.embeddings_generated == 50
        assert response.processing_time_s == 12.5
        assert response.errors == ["Failed to process doc3.pdf"]
        assert response.correlation_id == "ing-789"


class TestHealthCheckResponse:
    """Test HealthCheckResponse schema validation."""
    
    def test_health_check_response_valid(self):
        """Test valid health check response."""
        response = HealthCheckResponse(
            status="healthy",
            components={"embedder": "healthy", "llm": "healthy", "vector_store": "healthy"},
            timestamp=1609459200.0
        )
        
        assert response.status == "healthy"
        assert response.components == {"embedder": "healthy", "llm": "healthy", "vector_store": "healthy"}
        assert response.timestamp == 1609459200.0
    
    def test_health_check_response_degraded(self):
        """Test health check response with degraded status."""
        response = HealthCheckResponse(
            status="degraded",
            components={"embedder": "healthy", "llm": "unhealthy"},
            timestamp=1609459200.0
        )
        
        assert response.status == "degraded"
    
    def test_health_check_response_invalid_status(self):
        """Test health check response with invalid status."""
        with pytest.raises(ValidationError) as exc_info:
            HealthCheckResponse(
                status="unknown",
                components={},
                timestamp=1609459200.0
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("status",)
        assert "String should match pattern" in errors[0]["msg"]


class TestStatsResponse:
    """Test StatsResponse schema validation."""
    
    def test_stats_response_valid(self):
        """Test valid stats response."""
        response = StatsResponse(
            pipeline_config={"chunk_size": 1000, "top_k": 5},
            vector_store={"document_count": 100, "index_size": "10MB"},
            embedding_dimension=384
        )
        
        assert response.pipeline_config == {"chunk_size": 1000, "top_k": 5}
        assert response.vector_store == {"document_count": 100, "index_size": "10MB"}
        assert response.embedding_dimension == 384


class TestConfigUpdateRequest:
    """Test ConfigUpdateRequest schema validation."""
    
    def test_config_update_request_valid(self):
        """Test valid config update request."""
        request = ConfigUpdateRequest(
            config_overrides={"chunking": {"chunk_size": 1500}, "llm": {"temperature": 0.1}}
        )
        
        assert request.config_overrides == {"chunking": {"chunk_size": 1500}, "llm": {"temperature": 0.1}}


class TestConfigResponse:
    """Test ConfigResponse schema validation."""
    
    def test_config_response_valid(self):
        """Test valid config response."""
        response = ConfigResponse(
            current_config={"chunking": {"chunk_size": 1000}},
            applied_overrides={"chunking": {"chunk_size": 1500}}
        )
        
        assert response.current_config == {"chunking": {"chunk_size": 1000}}
        assert response.applied_overrides == {"chunking": {"chunk_size": 1500}}
    
    def test_config_response_minimal(self):
        """Test minimal config response."""
        response = ConfigResponse(
            current_config={"test": "value"}
        )
        
        assert response.applied_overrides is None


class TestStreamingQueryResponse:
    """Test StreamingQueryResponse schema validation."""
    
    def test_streaming_response_valid(self):
        """Test valid streaming response."""
        response = StreamingQueryResponse(
            chunk_type="answer",
            content="This is part of the answer",
            metadata={"chunk_index": 1},
            is_complete=False
        )
        
        assert response.chunk_type == "answer"
        assert response.content == "This is part of the answer"
        assert response.metadata == {"chunk_index": 1}
        assert response.is_complete is False
    
    def test_streaming_response_complete(self):
        """Test streaming response completion."""
        response = StreamingQueryResponse(
            chunk_type="complete",
            is_complete=True
        )
        
        assert response.chunk_type == "complete"
        assert response.content is None
        assert response.is_complete is True
    
    def test_streaming_response_invalid_chunk_type(self):
        """Test streaming response with invalid chunk type."""
        with pytest.raises(ValidationError) as exc_info:
            StreamingQueryResponse(chunk_type="invalid")
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("chunk_type",)
        assert "String should match pattern" in errors[0]["msg"]


class TestDocumentValidationRequest:
    """Test DocumentValidationRequest schema validation."""
    
    def test_document_validation_request_valid(self):
        """Test valid document validation request."""
        request = DocumentValidationRequest(
            document_path="/path/to/document.pdf",
            check_encoding=True,
            check_size=False
        )
        
        assert request.document_path == "/path/to/document.pdf"
        assert request.check_encoding is True
        assert request.check_size is False
    
    def test_document_validation_request_defaults(self):
        """Test document validation request with defaults."""
        request = DocumentValidationRequest(
            document_path="/path/to/doc.txt"
        )
        
        assert request.check_encoding is True
        assert request.check_size is True


class TestDocumentValidationResponse:
    """Test DocumentValidationResponse schema validation."""
    
    def test_document_validation_response_valid(self):
        """Test valid document validation response."""
        response = DocumentValidationResponse(
            is_valid=True,
            file_size_bytes=1024,
            encoding="utf-8",
            issues=[],
            supported=True
        )
        
        assert response.is_valid is True
        assert response.file_size_bytes == 1024
        assert response.encoding == "utf-8"
        assert response.issues == []
        assert response.supported is True
    
    def test_document_validation_response_with_issues(self):
        """Test document validation response with issues."""
        response = DocumentValidationResponse(
            is_valid=False,
            file_size_bytes=0,
            issues=["File is empty", "Unsupported encoding"],
            supported=False
        )
        
        assert response.is_valid is False
        assert response.encoding is None
        assert len(response.issues) == 2


class TestBatchQueryRequest:
    """Test BatchQueryRequest schema validation."""
    
    def test_batch_query_request_valid(self):
        """Test valid batch query request."""
        queries = [
            QueryRequest(question="Question 1"),
            QueryRequest(question="Question 2")
        ]
        
        request = BatchQueryRequest(
            queries=queries,
            parallel_processing=True
        )
        
        assert len(request.queries) == 2
        assert request.parallel_processing is True
    
    def test_batch_query_request_empty(self):
        """Test batch query request with empty queries."""
        with pytest.raises(ValidationError) as exc_info:
            BatchQueryRequest(queries=[])
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("queries",)
        assert "at least 1 item" in errors[0]["msg"]
    
    def test_batch_query_request_too_many(self):
        """Test batch query request with too many queries."""
        queries = [QueryRequest(question=f"Question {i}") for i in range(101)]
        
        with pytest.raises(ValidationError) as exc_info:
            BatchQueryRequest(queries=queries)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("queries",)
        assert "at most 100 items" in errors[0]["msg"]


class TestBatchQueryResponse:
    """Test BatchQueryResponse schema validation."""
    
    def test_batch_query_response_valid(self):
        """Test valid batch query response."""
        results = [
            QueryResponse(
                answer="Answer 1",
                correlation_id="corr-1",
                processing_time_ms=100,
                retrieved_chunks=1,
                model_info={"llm": "gpt-4"}
            )
        ]
        
        response = BatchQueryResponse(
            results=results,
            total_processing_time_ms=150,
            successful_queries=1,
            failed_queries=0
        )
        
        assert len(response.results) == 1
        assert response.total_processing_time_ms == 150
        assert response.successful_queries == 1
        assert response.failed_queries == 0


class TestSearchFilter:
    """Test SearchFilter schema validation."""
    
    def test_search_filter_valid(self):
        """Test valid search filter."""
        filter_obj = SearchFilter(
            field="source",
            operator="eq",
            value="document.pdf"
        )
        
        assert filter_obj.field == "source"
        assert filter_obj.operator == "eq"
        assert filter_obj.value == "document.pdf"
    
    def test_search_filter_invalid_operator(self):
        """Test search filter with invalid operator."""
        with pytest.raises(ValidationError) as exc_info:
            SearchFilter(
                field="source",
                operator="invalid",
                value="test"
            )
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("operator",)
        assert "String should match pattern" in errors[0]["msg"]


class TestAdvancedQueryRequest:
    """Test AdvancedQueryRequest schema validation."""
    
    def test_advanced_query_request_valid(self):
        """Test valid advanced query request."""
        filters = [
            SearchFilter(field="source", operator="eq", value="doc.pdf"),
            SearchFilter(field="score", operator="gte", value=0.8)
        ]
        
        request = AdvancedQueryRequest(
            question="Advanced question",
            filters=filters,
            top_k=10,
            score_threshold=0.8,
            rerank=True,
            explain_ranking=True
        )
        
        assert request.question == "Advanced question"
        assert len(request.filters) == 2
        assert request.top_k == 10
        assert request.score_threshold == 0.8
        assert request.rerank is True
        assert request.explain_ranking is True
    
    def test_advanced_query_request_defaults(self):
        """Test advanced query request with defaults."""
        request = AdvancedQueryRequest(question="Test question")
        
        assert request.filters == []
        assert request.top_k == 5
        assert request.score_threshold == 0.7
        assert request.rerank is False
        assert request.explain_ranking is False


class TestSystemCommand:
    """Test SystemCommand enum."""
    
    def test_system_command_values(self):
        """Test system command enum values."""
        assert SystemCommand.RELOAD_CONFIG == "reload_config"
        assert SystemCommand.CLEAR_CACHE == "clear_cache"
        assert SystemCommand.REBUILD_INDEX == "rebuild_index"
        assert SystemCommand.EXPORT_DATA == "export_data"
        assert SystemCommand.IMPORT_DATA == "import_data"


class TestAdminRequest:
    """Test AdminRequest schema validation."""
    
    def test_admin_request_valid(self):
        """Test valid admin request."""
        request = AdminRequest(
            command=SystemCommand.RELOAD_CONFIG,
            parameters={"config_path": "/new/config.yaml"},
            force=True
        )
        
        assert request.command == SystemCommand.RELOAD_CONFIG
        assert request.parameters == {"config_path": "/new/config.yaml"}
        assert request.force is True
    
    def test_admin_request_minimal(self):
        """Test minimal admin request."""
        request = AdminRequest(command=SystemCommand.CLEAR_CACHE)
        
        assert request.parameters is None
        assert request.force is False


class TestAdminResponse:
    """Test AdminResponse schema validation."""
    
    def test_admin_response_valid(self):
        """Test valid admin response."""
        response = AdminResponse(
            command="reload_config",
            status="success",
            message="Configuration reloaded successfully",
            details={"configs_loaded": 3},
            execution_time_ms=500
        )
        
        assert response.command == "reload_config"
        assert response.status == "success"
        assert response.message == "Configuration reloaded successfully"
        assert response.details == {"configs_loaded": 3}
        assert response.execution_time_ms == 500


class TestMetricsResponse:
    """Test MetricsResponse schema validation."""
    
    def test_metrics_response_valid(self):
        """Test valid metrics response."""
        response = MetricsResponse(
            metrics="# HELP test_metric Test metric\ntest_metric 1.0",
            format="prometheus",
            timestamp=1609459200.0
        )
        
        assert "test_metric" in response.metrics
        assert response.format == "prometheus"
        assert response.timestamp == 1609459200.0
    
    def test_metrics_response_default_format(self):
        """Test metrics response with default format."""
        response = MetricsResponse(
            metrics="test data",
            timestamp=1609459200.0
        )
        
        assert response.format == "prometheus"


class TestLogsRequest:
    """Test LogsRequest schema validation."""
    
    def test_logs_request_valid(self):
        """Test valid logs request."""
        request = LogsRequest(
            level="ERROR",
            since="2021-01-01T00:00:00Z",
            limit=500,
            correlation_id="corr-logs-123"
        )
        
        assert request.level == "ERROR"
        assert request.since == "2021-01-01T00:00:00Z"
        assert request.limit == 500
        assert request.correlation_id == "corr-logs-123"
    
    def test_logs_request_defaults(self):
        """Test logs request with defaults."""
        request = LogsRequest()
        
        assert request.level is None
        assert request.since is None
        assert request.limit == 100
        assert request.correlation_id is None
    
    def test_logs_request_invalid_level(self):
        """Test logs request with invalid level."""
        with pytest.raises(ValidationError) as exc_info:
            LogsRequest(level="INVALID")
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("level",)
        assert "String should match pattern" in errors[0]["msg"]
    
    def test_logs_request_invalid_limit(self):
        """Test logs request with invalid limit."""
        with pytest.raises(ValidationError) as exc_info:
            LogsRequest(limit=0)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("limit",)
        assert "greater than or equal to 1" in errors[0]["msg"]
        
        with pytest.raises(ValidationError) as exc_info:
            LogsRequest(limit=10001)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("limit",)
        assert "less than or equal to 10000" in errors[0]["msg"]


class TestLogEntry:
    """Test LogEntry schema validation."""
    
    def test_log_entry_valid(self):
        """Test valid log entry."""
        entry = LogEntry(
            timestamp="2021-01-01T12:00:00Z",
            level="INFO",
            logger="enhanced_rag.pipeline",
            message="Processing query",
            correlation_id="corr-123",
            metadata={"query_id": "q-456", "user": "test"}
        )
        
        assert entry.timestamp == "2021-01-01T12:00:00Z"
        assert entry.level == "INFO"
        assert entry.logger == "enhanced_rag.pipeline"
        assert entry.message == "Processing query"
        assert entry.correlation_id == "corr-123"
        assert entry.metadata == {"query_id": "q-456", "user": "test"}
    
    def test_log_entry_minimal(self):
        """Test minimal log entry."""
        entry = LogEntry(
            timestamp="2021-01-01T12:00:00Z",
            level="INFO",
            logger="test.logger",
            message="Test message"
        )
        
        assert entry.correlation_id is None
        assert entry.metadata is None


class TestLogsResponse:
    """Test LogsResponse schema validation."""
    
    def test_logs_response_valid(self):
        """Test valid logs response."""
        logs = [
            LogEntry(
                timestamp="2021-01-01T12:00:00Z",
                level="INFO",
                logger="test.logger",
                message="Log message 1"
            ),
            LogEntry(
                timestamp="2021-01-01T12:01:00Z",
                level="ERROR",
                logger="test.logger",
                message="Log message 2"
            )
        ]
        
        response = LogsResponse(
            logs=logs,
            total_entries=100,
            has_more=True
        )
        
        assert len(response.logs) == 2
        assert response.total_entries == 100
        assert response.has_more is True
    
    def test_logs_response_empty(self):
        """Test empty logs response."""
        response = LogsResponse(
            total_entries=0,
            has_more=False
        )
        
        assert response.logs == []
        assert response.total_entries == 0
        assert response.has_more is False


class TestSchemaIntegration:
    """Integration tests for schema validation."""
    
    def test_query_to_response_workflow(self):
        """Test complete query workflow from request to response."""
        # Create query request
        request = QueryRequest(
            question="What is the capital of France?",
            top_k=3,
            include_metadata=True
        )
        
        # Create source info
        sources = [
            SourceInfo(
                chunk_id="chunk_1",
                content="Paris is the capital of France.",
                score=0.95,
                metadata={"source": "geography.pdf", "page": 12}
            )
        ]
        
        # Create response
        response = QueryResponse(
            answer="The capital of France is Paris.",
            correlation_id="workflow-test-123",
            processing_time_ms=850,
            retrieved_chunks=1,
            model_info={"embedder": "text-ada-002", "llm": "gpt-4"},
            sources=sources
        )
        
        # Validate the workflow
        assert request.question == "What is the capital of France?"
        assert response.answer == "The capital of France is Paris."
        assert len(response.sources) == 1
        assert response.sources[0].score == 0.95
    
    def test_error_handling_workflow(self):
        """Test error response creation."""
        try:
            # This would simulate a validation error
            QueryRequest(question="")
        except ValidationError as e:
            error_response = ErrorResponse(
                error="Validation failed",
                error_type="ValidationError",
                correlation_id="error-test-456",
                details={"validation_errors": str(e)}
            )
            
            assert error_response.error_type == "ValidationError"
            assert error_response.correlation_id == "error-test-456"
            assert "validation_errors" in error_response.details
    
    def test_batch_processing_workflow(self):
        """Test batch processing schemas."""
        # Create batch request
        queries = [
            QueryRequest(question="Question 1"),
            QueryRequest(question="Question 2"),
            QueryRequest(question="Question 3")
        ]
        
        batch_request = BatchQueryRequest(
            queries=queries,
            parallel_processing=True
        )
        
        # Create batch response
        results = [
            QueryResponse(
                answer=f"Answer {i}",
                correlation_id=f"batch-{i}",
                processing_time_ms=100,
                retrieved_chunks=1,
                model_info={"llm": "gpt-4"}
            )
            for i in range(3)
        ]
        
        batch_response = BatchQueryResponse(
            results=results,
            total_processing_time_ms=350,
            successful_queries=3,
            failed_queries=0
        )
        
        assert len(batch_request.queries) == 3
        assert len(batch_response.results) == 3
        assert batch_response.successful_queries == 3
        assert batch_response.failed_queries == 0