"""Unit tests for metrics utilities."""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from prometheus_client import CollectorRegistry, REGISTRY
from prometheus_client.parser import text_string_to_metric_families

from enhanced_rag.utils.metrics import (
    RAGMetrics,
    time_operation,
    RequestTracker,
    MetricsServer,
    get_metrics,
    setup_metrics
)


class TestRAGMetrics:
    """Test RAGMetrics functionality."""
    
    def setup_method(self):
        """Setup test metrics with isolated registry."""
        self.registry = CollectorRegistry()
        self.metrics = RAGMetrics(registry=self.registry)
    
    def test_rag_metrics_init_default_registry(self):
        """Test RAGMetrics initialization with default registry."""
        metrics = RAGMetrics()
        assert metrics.registry is not None
    
    def test_rag_metrics_init_custom_registry(self):
        """Test RAGMetrics initialization with custom registry."""
        custom_registry = CollectorRegistry()
        metrics = RAGMetrics(registry=custom_registry)
        assert metrics.registry == custom_registry
    
    def test_metrics_setup(self):
        """Test that all metrics are properly initialized."""
        # Check that key metrics exist
        assert hasattr(self.metrics, 'documents_processed')
        assert hasattr(self.metrics, 'chunks_created')
        assert hasattr(self.metrics, 'ingestion_duration')
        assert hasattr(self.metrics, 'embeddings_generated')
        assert hasattr(self.metrics, 'embedding_duration')
        assert hasattr(self.metrics, 'vector_store_operations')
        assert hasattr(self.metrics, 'queries_processed')
        assert hasattr(self.metrics, 'llm_calls')
        assert hasattr(self.metrics, 'errors')
        assert hasattr(self.metrics, 'system_health')
    
    def test_record_document_processed(self):
        """Test recording processed documents."""
        self.metrics.record_document_processed("success")
        self.metrics.record_document_processed("error")
        self.metrics.record_document_processed("success")
        
        metrics_output = self.metrics.get_metrics()
        
        # Check that metrics are recorded
        assert 'rag_documents_processed_total{status="success"} 2.0' in metrics_output
        assert 'rag_documents_processed_total{status="error"} 1.0' in metrics_output
    
    def test_record_chunks_created(self):
        """Test recording created chunks."""
        self.metrics.record_chunks_created(5, "recursive")
        self.metrics.record_chunks_created(3, "sentence")
        self.metrics.record_chunks_created(2, "recursive")
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_chunks_created_total{strategy="recursive"} 7.0' in metrics_output
        assert 'rag_chunks_created_total{strategy="sentence"} 3.0' in metrics_output
    
    def test_record_ingestion_time(self):
        """Test recording ingestion time."""
        self.metrics.record_ingestion_time(1.5, 10)
        self.metrics.record_ingestion_time(2.3, 20)
        
        metrics_output = self.metrics.get_metrics()
        
        # Check that histogram metrics are present
        assert 'rag_ingestion_duration_seconds_bucket' in metrics_output
        assert 'batch_size="10"' in metrics_output
        assert 'batch_size="20"' in metrics_output
    
    def test_record_embeddings_generated(self):
        """Test recording generated embeddings."""
        self.metrics.record_embeddings_generated(100, "text-ada-002", "document")
        self.metrics.record_embeddings_generated(1, "text-ada-002", "query")
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_embeddings_generated_total{model="text-ada-002",type="document"} 100.0' in metrics_output
        assert 'rag_embeddings_generated_total{model="text-ada-002",type="query"} 1.0' in metrics_output
    
    def test_record_embedding_time(self):
        """Test recording embedding time."""
        self.metrics.record_embedding_time(0.5, "text-ada-002", 50)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_embedding_duration_seconds_bucket' in metrics_output
        assert 'model="text-ada-002"' in metrics_output
        assert 'batch_size="50"' in metrics_output
    
    def test_record_embedding_tokens(self):
        """Test recording embedding tokens."""
        self.metrics.record_embedding_tokens(1000, "text-ada-002")
        self.metrics.record_embedding_tokens(500, "text-ada-002")
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_embedding_tokens_total{model="text-ada-002"} 1500.0' in metrics_output
    
    def test_record_vector_operation(self):
        """Test recording vector store operations."""
        self.metrics.record_vector_operation("search", "success", 0.05)
        self.metrics.record_vector_operation("insert", "error", 0.02)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_vector_store_operations_total{operation="search",status="success"} 1.0' in metrics_output
        assert 'rag_vector_store_operations_total{operation="insert",status="error"} 1.0' in metrics_output
        assert 'rag_vector_store_duration_seconds_bucket' in metrics_output
    
    def test_update_vector_store_size(self):
        """Test updating vector store size."""
        self.metrics.update_vector_store_size(1000)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_vector_store_documents_count 1000.0' in metrics_output
    
    def test_record_query_processed(self):
        """Test recording processed queries."""
        self.metrics.record_query_processed("success", 1.2, True)
        self.metrics.record_query_processed("error", 0.8, False)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_queries_processed_total{status="success"} 1.0' in metrics_output
        assert 'rag_queries_processed_total{status="error"} 1.0' in metrics_output
        assert 'rag_query_duration_seconds_bucket' in metrics_output
        assert 'has_results="True"' in metrics_output
        assert 'has_results="False"' in metrics_output
    
    def test_record_retrieved_chunks(self):
        """Test recording retrieved chunks."""
        self.metrics.record_retrieved_chunks(5)
        self.metrics.record_retrieved_chunks(10)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_retrieved_chunks_count_bucket' in metrics_output
    
    def test_record_similarity_scores(self):
        """Test recording similarity scores."""
        scores = [0.9, 0.8, 0.7, 0.6]
        self.metrics.record_similarity_scores(scores)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_similarity_scores_bucket' in metrics_output
        assert 'rag_similarity_scores_count 4.0' in metrics_output
    
    def test_record_llm_call(self):
        """Test recording LLM calls."""
        self.metrics.record_llm_call("gpt-4", "success", 2.5)
        self.metrics.record_llm_call("gpt-3.5-turbo", "error", 1.0)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_llm_calls_total{model="gpt-4",status="success"} 1.0' in metrics_output
        assert 'rag_llm_calls_total{model="gpt-3.5-turbo",status="error"} 1.0' in metrics_output
        assert 'rag_llm_duration_seconds_bucket' in metrics_output
    
    def test_record_llm_tokens(self):
        """Test recording LLM tokens."""
        self.metrics.record_llm_tokens(100, 50, "gpt-4")
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_llm_tokens_total{model="gpt-4",type="prompt"} 100.0' in metrics_output
        assert 'rag_llm_tokens_total{model="gpt-4",type="completion"} 50.0' in metrics_output
    
    def test_update_llm_token_rate(self):
        """Test updating LLM token rate."""
        self.metrics.update_llm_token_rate(150.5, "gpt-4")
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_llm_tokens_per_second{model="gpt-4"} 150.5' in metrics_output
    
    def test_record_cache_operation(self):
        """Test recording cache operations."""
        self.metrics.record_cache_operation("get", "hit")
        self.metrics.record_cache_operation("get", "miss")
        self.metrics.record_cache_operation("set", "success")
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_cache_operations_total{operation="get",result="hit"} 1.0' in metrics_output
        assert 'rag_cache_operations_total{operation="get",result="miss"} 1.0' in metrics_output
        assert 'rag_cache_operations_total{operation="set",result="success"} 1.0' in metrics_output
    
    def test_update_cache_hit_rate(self):
        """Test updating cache hit rate."""
        self.metrics.update_cache_hit_rate(85.5, "embedding")
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_cache_hit_rate{cache_type="embedding"} 85.5' in metrics_output
    
    def test_record_error(self):
        """Test recording errors."""
        self.metrics.record_error("embedder", "ConnectionError")
        self.metrics.record_error("llm", "RateLimitError")
        self.metrics.record_error("embedder", "ConnectionError")
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_errors_total{component="embedder",error_type="ConnectionError"} 2.0' in metrics_output
        assert 'rag_errors_total{component="llm",error_type="RateLimitError"} 1.0' in metrics_output
    
    def test_record_retry_attempt(self):
        """Test recording retry attempts."""
        self.metrics.record_retry_attempt("llm_call", 1)
        self.metrics.record_retry_attempt("llm_call", 2)
        self.metrics.record_retry_attempt("embedding", 1)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_retry_attempts_total{attempt_number="1",operation="llm_call"} 1.0' in metrics_output
        assert 'rag_retry_attempts_total{attempt_number="2",operation="llm_call"} 1.0' in metrics_output
        assert 'rag_retry_attempts_total{attempt_number="1",operation="embedding"} 1.0' in metrics_output
    
    def test_update_active_requests(self):
        """Test updating active requests."""
        self.metrics.update_active_requests(5)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_active_requests 5.0' in metrics_output
    
    def test_update_system_health(self):
        """Test updating system health."""
        self.metrics.update_system_health("embedder", True)
        self.metrics.update_system_health("llm", False)
        
        metrics_output = self.metrics.get_metrics()
        
        assert 'rag_system_health{component="embedder"} 1.0' in metrics_output
        assert 'rag_system_health{component="llm"} 0.0' in metrics_output
    
    def test_get_metrics_format(self):
        """Test metrics output format."""
        self.metrics.record_document_processed("success")
        output = self.metrics.get_metrics()
        
        assert isinstance(output, str)
        assert output.startswith("# HELP")
        assert "rag_documents_processed_total" in output


class TestTimeOperationDecorator:
    """Test time_operation decorator functionality."""
    
    def setup_method(self):
        """Setup test metrics."""
        self.registry = CollectorRegistry()
        self.metrics = RAGMetrics(registry=self.registry)
    
    @pytest.mark.asyncio
    async def test_time_operation_async_success(self):
        """Test timing async operation success."""
        
        @time_operation(self.metrics, "query")
        async def mock_query():
            await asyncio.sleep(0.01)  # Small delay
            return "success"
        
        result = await mock_query()
        
        assert result == "success"
        
        metrics_output = self.metrics.get_metrics()
        assert 'rag_queries_processed_total{status="success"}' in metrics_output
        assert 'rag_query_duration_seconds_bucket' in metrics_output
    
    @pytest.mark.asyncio
    async def test_time_operation_async_error(self):
        """Test timing async operation error."""
        
        @time_operation(self.metrics, "query")
        async def mock_failing_query():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await mock_failing_query()
        
        metrics_output = self.metrics.get_metrics()
        assert 'rag_queries_processed_total{status="error"}' in metrics_output
        assert 'rag_errors_total{component="query",error_type="ValueError"}' in metrics_output
    
    @pytest.mark.asyncio
    async def test_time_operation_embedding(self):
        """Test timing embedding operation."""
        
        @time_operation(self.metrics, "embedding", model="test-model", batch_size=10)
        async def mock_embedding():
            await asyncio.sleep(0.01)
            return "embeddings"
        
        result = await mock_embedding()
        
        assert result == "embeddings"
        
        metrics_output = self.metrics.get_metrics()
        assert 'rag_embedding_duration_seconds_bucket' in metrics_output
        assert 'model="test-model"' in metrics_output
        assert 'batch_size="10"' in metrics_output
    
    @pytest.mark.asyncio
    async def test_time_operation_llm(self):
        """Test timing LLM operation."""
        
        @time_operation(self.metrics, "llm", model="gpt-4")
        async def mock_llm_call():
            await asyncio.sleep(0.01)
            return "response"
        
        result = await mock_llm_call()
        
        assert result == "response"
        
        metrics_output = self.metrics.get_metrics()
        assert 'rag_llm_calls_total{model="gpt-4",status="success"}' in metrics_output
        assert 'rag_llm_duration_seconds_bucket' in metrics_output
    
    def test_time_operation_sync_success(self):
        """Test timing sync operation success."""
        
        @time_operation(self.metrics, "ingestion", batch_size=5)
        def mock_ingestion():
            time.sleep(0.01)
            return "ingested"
        
        result = mock_ingestion()
        
        assert result == "ingested"
        
        metrics_output = self.metrics.get_metrics()
        assert 'rag_ingestion_duration_seconds_bucket' in metrics_output
        assert 'batch_size="5"' in metrics_output
    
    def test_time_operation_sync_error(self):
        """Test timing sync operation error."""
        
        @time_operation(self.metrics, "ingestion", batch_size=5)
        def mock_failing_ingestion():
            time.sleep(0.01)
            raise RuntimeError("Ingestion failed")
        
        with pytest.raises(RuntimeError, match="Ingestion failed"):
            mock_failing_ingestion()
        
        metrics_output = self.metrics.get_metrics()
        assert 'rag_errors_total{component="ingestion",error_type="RuntimeError"}' in metrics_output


class TestRequestTracker:
    """Test RequestTracker context manager."""
    
    def setup_method(self):
        """Setup test metrics."""
        self.registry = CollectorRegistry()
        self.metrics = RAGMetrics(registry=self.registry)
        self.tracker = RequestTracker(self.metrics)
    
    def test_request_tracker_context_manager(self):
        """Test RequestTracker as context manager."""
        # Initially no active requests
        assert self.tracker._active_count == 0
        
        with self.tracker:
            assert self.tracker._active_count == 1
            
            # Check metrics
            metrics_output = self.metrics.get_metrics()
            assert 'rag_active_requests 1.0' in metrics_output
        
        # After context, count should be back to 0
        assert self.tracker._active_count == 0
        
        metrics_output = self.metrics.get_metrics()
        assert 'rag_active_requests 0.0' in metrics_output
    
    def test_request_tracker_nested_contexts(self):
        """Test nested RequestTracker contexts."""
        with self.tracker:
            assert self.tracker._active_count == 1
            
            with self.tracker:
                assert self.tracker._active_count == 2
                
                metrics_output = self.metrics.get_metrics()
                assert 'rag_active_requests 2.0' in metrics_output
            
            assert self.tracker._active_count == 1
        
        assert self.tracker._active_count == 0
    
    def test_request_tracker_with_exception(self):
        """Test RequestTracker cleanup when exception occurs."""
        try:
            with self.tracker:
                assert self.tracker._active_count == 1
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Count should still be reset after exception
        assert self.tracker._active_count == 0


class TestMetricsServer:
    """Test MetricsServer functionality."""
    
    def setup_method(self):
        """Setup test metrics and server."""
        self.registry = CollectorRegistry()
        self.metrics = RAGMetrics(registry=self.registry)
    
    @patch('enhanced_rag.utils.metrics.start_http_server')
    def test_metrics_server_start(self, mock_start_server):
        """Test starting metrics server."""
        mock_server = Mock()
        mock_start_server.return_value = mock_server
        
        server = MetricsServer(self.metrics, port=8000)
        server.start()
        
        mock_start_server.assert_called_once_with(8000, registry=self.metrics.registry)
        assert server.server == mock_server
    
    @patch('enhanced_rag.utils.metrics.start_http_server')
    def test_metrics_server_start_error(self, mock_start_server):
        """Test metrics server start error handling."""
        mock_start_server.side_effect = Exception("Port already in use")
        
        server = MetricsServer(self.metrics, port=8000)
        
        with pytest.raises(Exception, match="Port already in use"):
            server.start()
    
    def test_metrics_server_stop(self):
        """Test stopping metrics server."""
        mock_server = Mock()
        
        server = MetricsServer(self.metrics, port=8000)
        server.server = mock_server
        
        server.stop()
        
        mock_server.shutdown.assert_called_once()
    
    def test_metrics_server_stop_no_server(self):
        """Test stopping metrics server when no server is running."""
        server = MetricsServer(self.metrics, port=8000)
        
        # Should not raise exception
        server.stop()


class TestGlobalMetrics:
    """Test global metrics functionality."""
    
    def teardown_method(self):
        """Clean up global metrics."""
        import enhanced_rag.utils.metrics
        enhanced_rag.utils.metrics._global_metrics = None
    
    def test_get_metrics_creates_instance(self):
        """Test that get_metrics creates instance if none exists."""
        metrics = get_metrics()
        
        assert isinstance(metrics, RAGMetrics)
        assert metrics.registry is not None
    
    def test_get_metrics_returns_same_instance(self):
        """Test that get_metrics returns same instance on subsequent calls."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        
        assert metrics1 is metrics2
    
    @patch('enhanced_rag.utils.metrics.MetricsServer')
    def test_setup_metrics_without_port(self, mock_server_class):
        """Test setup_metrics without port."""
        metrics = setup_metrics()
        
        assert isinstance(metrics, RAGMetrics)
        mock_server_class.assert_not_called()
    
    @patch('enhanced_rag.utils.metrics.MetricsServer')
    def test_setup_metrics_with_port(self, mock_server_class):
        """Test setup_metrics with port."""
        mock_server = Mock()
        mock_server_class.return_value = mock_server
        
        metrics = setup_metrics(port=8000)
        
        assert isinstance(metrics, RAGMetrics)
        mock_server_class.assert_called_once_with(metrics, 8000)
        mock_server.start.assert_called_once()


class TestMetricsIntegration:
    """Integration tests for metrics components."""
    
    def setup_method(self):
        """Setup test metrics."""
        self.registry = CollectorRegistry()
        self.metrics = RAGMetrics(registry=self.registry)
    
    def test_complete_pipeline_metrics(self):
        """Test recording metrics for a complete pipeline operation."""
        # Record document processing
        self.metrics.record_document_processed("success")
        self.metrics.record_chunks_created(10, "recursive")
        self.metrics.record_ingestion_time(2.5, 1)
        
        # Record embedding generation
        self.metrics.record_embeddings_generated(10, "text-ada-002", "document")
        self.metrics.record_embedding_time(1.2, "text-ada-002", 10)
        self.metrics.record_embedding_tokens(1000, "text-ada-002")
        
        # Record vector store operations
        self.metrics.record_vector_operation("insert", "success", 0.5)
        self.metrics.update_vector_store_size(100)
        
        # Record query processing
        self.metrics.record_query_processed("success", 1.8, True)
        self.metrics.record_retrieved_chunks(5)
        self.metrics.record_similarity_scores([0.9, 0.8, 0.7, 0.6, 0.5])
        
        # Record LLM call
        self.metrics.record_llm_call("gpt-4", "success", 2.1)
        self.metrics.record_llm_tokens(150, 80, "gpt-4")
        
        # Update system health
        self.metrics.update_system_health("embedder", True)
        self.metrics.update_system_health("llm", True)
        
        # Get all metrics
        metrics_output = self.metrics.get_metrics()
        
        # Verify key metrics are present
        assert 'rag_documents_processed_total{status="success"} 1.0' in metrics_output
        assert 'rag_chunks_created_total{strategy="recursive"} 10.0' in metrics_output
        assert 'rag_embeddings_generated_total' in metrics_output
        assert 'rag_vector_store_documents_count 100.0' in metrics_output
        assert 'rag_queries_processed_total{status="success"} 1.0' in metrics_output
        assert 'rag_llm_calls_total{model="gpt-4",status="success"} 1.0' in metrics_output
        assert 'rag_system_health{component="embedder"} 1.0' in metrics_output
    
    @pytest.mark.asyncio
    async def test_metrics_with_decorators_and_context_managers(self):
        """Test metrics with decorators and context managers."""
        
        @time_operation(self.metrics, "query")
        async def mock_query_operation():
            await asyncio.sleep(0.01)
            return "result"
        
        # Use request tracker
        with RequestTracker(self.metrics):
            result = await mock_query_operation()
        
        assert result == "result"
        
        metrics_output = self.metrics.get_metrics()
        
        # Check that both decorator and context manager recorded metrics
        assert 'rag_queries_processed_total{status="success"}' in metrics_output
        assert 'rag_active_requests 0.0' in metrics_output  # Should be 0 after context
    
    def test_error_tracking_across_components(self):
        """Test error tracking across different components."""
        # Record various errors
        self.metrics.record_error("embedder", "ConnectionError")
        self.metrics.record_error("llm", "RateLimitError")
        self.metrics.record_error("vector_store", "IndexError")
        self.metrics.record_error("embedder", "ConnectionError")  # Duplicate
        
        # Record retry attempts
        self.metrics.record_retry_attempt("llm_call", 1)
        self.metrics.record_retry_attempt("llm_call", 2)
        self.metrics.record_retry_attempt("embedding", 1)
        
        metrics_output = self.metrics.get_metrics()
        
        # Verify error counts
        assert 'rag_errors_total{component="embedder",error_type="ConnectionError"} 2.0' in metrics_output
        assert 'rag_errors_total{component="llm",error_type="RateLimitError"} 1.0' in metrics_output
        assert 'rag_errors_total{component="vector_store",error_type="IndexError"} 1.0' in metrics_output
        
        # Verify retry counts
        assert 'rag_retry_attempts_total{attempt_number="1",operation="llm_call"} 1.0' in metrics_output
        assert 'rag_retry_attempts_total{attempt_number="2",operation="llm_call"} 1.0' in metrics_output
    
    def test_performance_tracking(self):
        """Test comprehensive performance tracking."""
        # Record various timing metrics
        self.metrics.record_ingestion_time(5.0, 50)
        self.metrics.record_embedding_time(2.0, "text-ada-002", 100)
        self.metrics.record_query_processed("success", 1.5, True)
        self.metrics.record_llm_call("gpt-4", "success", 3.0)
        
        # Record token usage and rates
        self.metrics.record_embedding_tokens(5000, "text-ada-002")
        self.metrics.record_llm_tokens(200, 100, "gpt-4")
        self.metrics.update_llm_token_rate(150.0, "gpt-4")
        
        metrics_output = self.metrics.get_metrics()
        
        # Verify timing histograms
        assert 'rag_ingestion_duration_seconds_bucket' in metrics_output
        assert 'rag_embedding_duration_seconds_bucket' in metrics_output
        assert 'rag_query_duration_seconds_bucket' in metrics_output
        assert 'rag_llm_duration_seconds_bucket' in metrics_output
        
        # Verify token metrics
        assert 'rag_embedding_tokens_total{model="text-ada-002"} 5000.0' in metrics_output
        assert 'rag_llm_tokens_total{model="gpt-4",type="prompt"} 200.0' in metrics_output
        assert 'rag_llm_tokens_total{model="gpt-4",type="completion"} 100.0' in metrics_output
        assert 'rag_llm_tokens_per_second{model="gpt-4"} 150.0' in metrics_output