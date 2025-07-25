"""Prometheus metrics for the RAG system."""

import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest,
    start_http_server, CONTENT_TYPE_LATEST
)
import logging

logger = logging.getLogger(__name__)


class RAGMetrics:
    """Centralized metrics collection for the RAG system."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # Document ingestion metrics
        self.documents_processed = Counter(
            'rag_documents_processed_total',
            'Total number of documents processed',
            ['status'],
            registry=self.registry
        )
        
        self.chunks_created = Counter(
            'rag_chunks_created_total',
            'Total number of text chunks created',
            ['strategy'],
            registry=self.registry
        )
        
        self.ingestion_duration = Histogram(
            'rag_ingestion_duration_seconds',
            'Time spent ingesting documents',
            ['batch_size'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        # Embedding metrics
        self.embeddings_generated = Counter(
            'rag_embeddings_generated_total',
            'Total number of embeddings generated',
            ['model', 'type'],
            registry=self.registry
        )
        
        self.embedding_duration = Histogram(
            'rag_embedding_duration_seconds',
            'Time spent generating embeddings',
            ['model', 'batch_size'],
            registry=self.registry,
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.embedding_tokens = Counter(
            'rag_embedding_tokens_total',
            'Total tokens processed for embeddings',
            ['model'],
            registry=self.registry
        )
        
        # Vector store metrics
        self.vector_store_operations = Counter(
            'rag_vector_store_operations_total',
            'Total vector store operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.vector_store_duration = Histogram(
            'rag_vector_store_duration_seconds',
            'Time spent on vector store operations',
            ['operation'],
            registry=self.registry,
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        self.vector_store_size = Gauge(
            'rag_vector_store_documents_count',
            'Number of documents in vector store',
            registry=self.registry
        )
        
        # Query processing metrics
        self.queries_processed = Counter(
            'rag_queries_processed_total',
            'Total number of queries processed',
            ['status'],
            registry=self.registry
        )
        
        self.query_duration = Histogram(
            'rag_query_duration_seconds',
            'Time spent processing queries',
            ['has_results'],
            registry=self.registry,
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.retrieved_chunks = Histogram(
            'rag_retrieved_chunks_count',
            'Number of chunks retrieved per query',
            registry=self.registry,
            buckets=[0, 1, 2, 5, 10, 20, 50, 100]
        )
        
        self.similarity_scores = Histogram(
            'rag_similarity_scores',
            'Distribution of similarity scores',
            registry=self.registry,
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # LLM metrics
        self.llm_calls = Counter(
            'rag_llm_calls_total',
            'Total number of LLM API calls',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.llm_duration = Histogram(
            'rag_llm_duration_seconds',
            'Time spent on LLM calls',
            ['model'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.llm_tokens = Counter(
            'rag_llm_tokens_total',
            'Total tokens used in LLM calls',
            ['model', 'type'],
            registry=self.registry
        )
        
        self.llm_token_rate = Gauge(
            'rag_llm_tokens_per_second',
            'Current LLM token processing rate',
            ['model'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations = Counter(
            'rag_cache_operations_total',
            'Total cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'rag_cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        self.retry_attempts = Counter(
            'rag_retry_attempts_total',
            'Total number of retry attempts',
            ['operation', 'attempt_number'],
            registry=self.registry
        )
        
        # System metrics
        self.active_requests = Gauge(
            'rag_active_requests',
            'Number of currently active requests',
            registry=self.registry
        )
        
        self.system_health = Gauge(
            'rag_system_health',
            'System health status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )
    
    # Document ingestion methods
    def record_document_processed(self, status: str):
        """Record a processed document."""
        self.documents_processed.labels(status=status).inc()
    
    def record_chunks_created(self, count: int, strategy: str):
        """Record created chunks."""
        self.chunks_created.labels(strategy=strategy).inc(count)
    
    def record_ingestion_time(self, duration: float, batch_size: int):
        """Record document ingestion time."""
        self.ingestion_duration.labels(batch_size=str(batch_size)).observe(duration)
    
    # Embedding methods
    def record_embeddings_generated(self, count: int, model: str, embedding_type: str):
        """Record generated embeddings."""
        self.embeddings_generated.labels(model=model, type=embedding_type).inc(count)
    
    def record_embedding_time(self, duration: float, model: str, batch_size: int):
        """Record embedding generation time."""
        self.embedding_duration.labels(
            model=model, 
            batch_size=str(batch_size)
        ).observe(duration)
    
    def record_embedding_tokens(self, count: int, model: str):
        """Record embedding token usage."""
        self.embedding_tokens.labels(model=model).inc(count)
    
    # Vector store methods
    def record_vector_operation(self, operation: str, status: str, duration: float):
        """Record vector store operation."""
        self.vector_store_operations.labels(operation=operation, status=status).inc()
        self.vector_store_duration.labels(operation=operation).observe(duration)
    
    def update_vector_store_size(self, size: int):
        """Update vector store size."""
        self.vector_store_size.set(size)
    
    # Query processing methods
    def record_query_processed(self, status: str, duration: float, has_results: bool):
        """Record processed query."""
        self.queries_processed.labels(status=status).inc()
        self.query_duration.labels(has_results=str(has_results)).observe(duration)
    
    def record_retrieved_chunks(self, count: int):
        """Record number of retrieved chunks."""
        self.retrieved_chunks.observe(count)
    
    def record_similarity_scores(self, scores: list[float]):
        """Record similarity scores."""
        for score in scores:
            self.similarity_scores.observe(score)
    
    # LLM methods
    def record_llm_call(self, model: str, status: str, duration: float):
        """Record LLM API call."""
        self.llm_calls.labels(model=model, status=status).inc()
        self.llm_duration.labels(model=model).observe(duration)
    
    def record_llm_tokens(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Record LLM token usage."""
        self.llm_tokens.labels(model=model, type="prompt").inc(prompt_tokens)
        self.llm_tokens.labels(model=model, type="completion").inc(completion_tokens)
    
    def update_llm_token_rate(self, rate: float, model: str):
        """Update LLM token processing rate."""
        self.llm_token_rate.labels(model=model).set(rate)
    
    # Cache methods
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation."""
        self.cache_operations.labels(operation=operation, result=result).inc()
    
    def update_cache_hit_rate(self, rate: float, cache_type: str):
        """Update cache hit rate."""
        self.cache_hit_rate.labels(cache_type=cache_type).set(rate)
    
    # Error methods
    def record_error(self, component: str, error_type: str):
        """Record an error."""
        self.errors.labels(component=component, error_type=error_type).inc()
    
    def record_retry_attempt(self, operation: str, attempt_number: int):
        """Record retry attempt."""
        self.retry_attempts.labels(
            operation=operation, 
            attempt_number=str(attempt_number)
        ).inc()
    
    # System methods
    def update_active_requests(self, count: int):
        """Update active request count."""
        self.active_requests.set(count)
    
    def update_system_health(self, component: str, is_healthy: bool):
        """Update system health status."""
        self.system_health.labels(component=component).set(1 if is_healthy else 0)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


# Decorator for timing operations
def time_operation(metrics: RAGMetrics, operation_type: str, **labels):
    """Decorator to time operations and record metrics."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                metrics.record_error(operation_type, type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                
                if operation_type == "query":
                    has_results = kwargs.get('has_results', True)
                    metrics.record_query_processed(status, duration, has_results)
                elif operation_type == "embedding":
                    model = labels.get('model', 'unknown')
                    batch_size = labels.get('batch_size', 1)
                    metrics.record_embedding_time(duration, model, batch_size)
                elif operation_type == "llm":
                    model = labels.get('model', 'unknown')
                    metrics.record_llm_call(model, status, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                metrics.record_error(operation_type, type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                
                if operation_type == "ingestion":
                    batch_size = labels.get('batch_size', 1)
                    metrics.record_ingestion_time(duration, batch_size)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Context manager for tracking active requests
class RequestTracker:
    """Context manager for tracking active requests."""
    
    def __init__(self, metrics: RAGMetrics):
        self.metrics = metrics
        self._active_count = 0
    
    def __enter__(self):
        self._active_count += 1
        self.metrics.update_active_requests(self._active_count)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._active_count -= 1
        self.metrics.update_active_requests(self._active_count)


# Metrics server
class MetricsServer:
    """HTTP server for exposing Prometheus metrics."""
    
    def __init__(self, metrics: RAGMetrics, port: int = 8000):
        self.metrics = metrics
        self.port = port
        self.server = None
    
    def start(self):
        """Start the metrics server."""
        try:
            self.server = start_http_server(self.port, registry=self.metrics.registry)
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def stop(self):
        """Stop the metrics server."""
        if self.server:
            self.server.shutdown()
            logger.info("Metrics server stopped")


# Global metrics instance
_global_metrics: Optional[RAGMetrics] = None


def get_metrics() -> RAGMetrics:
    """Get or create global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = RAGMetrics()
    return _global_metrics


def setup_metrics(port: Optional[int] = None) -> RAGMetrics:
    """Setup metrics collection and optionally start server."""
    metrics = get_metrics()
    
    if port:
        server = MetricsServer(metrics, port)
        server.start()
    
    return metrics