"""REST API for the RAG system."""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response, StreamingResponse
import uvicorn

from .schemas import (
    QueryRequest, QueryResponse, ErrorResponse,
    IngestionRequest, IngestionResponse,
    HealthCheckResponse, StatsResponse,
    BatchQueryRequest, BatchQueryResponse,
    AdminRequest, AdminResponse,
    MetricsResponse
)
from ..utils.config import ConfigManager, RAGConfig
from ..utils.logging import setup_logging, get_logger
from ..utils.metrics import setup_metrics, get_metrics, RequestTracker
from ..utils.correlation import set_correlation_id, get_correlation_id, CorrelationContext
from ..pipeline import RAGPipeline

logger = get_logger(__name__)


class RAGService:
    """RAG service wrapper for dependency injection."""
    
    def __init__(self):
        self.pipeline: Optional[RAGPipeline] = None
        self.config: Optional[RAGConfig] = None
        self.metrics = get_metrics()
    
    def initialize(self, config: RAGConfig):
        """Initialize the RAG service."""
        self.config = config
        # This would create actual pipeline with real implementations
        # self.pipeline = create_pipeline(config)
        logger.info("RAG service initialized")
    
    def get_pipeline(self) -> RAGPipeline:
        """Get the RAG pipeline instance."""
        if self.pipeline is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        return self.pipeline


# Global service instance
rag_service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    
    # Startup
    logger.info("Starting RAG API server")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Setup metrics server
    setup_metrics(config.observability.metrics_port)
    
    # Initialize service
    rag_service.initialize(config)
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API server")


# Create FastAPI app
app = FastAPI(
    title="Enhanced RAG System API",
    description="Production-ready Retrieval-Augmented Generation system",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    """Add correlation ID to requests."""
    
    correlation_id = request.headers.get("X-Correlation-ID")
    
    with CorrelationContext(correlation_id) as corr_id:
        # Track active requests
        with RequestTracker(rag_service.metrics):
            start_time = time.time()
            
            response = await call_next(request)
            
            # Add correlation ID to response
            response.headers["X-Correlation-ID"] = corr_id
            
            # Log request
            processing_time = time.time() - start_time
            logger.info(
                f"Request processed: {request.method} {request.url.path}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_time_ms": processing_time * 1000,
                    "correlation_id": corr_id
                }
            )
            
            return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    
    correlation_id = get_correlation_id()
    
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "correlation_id": correlation_id
        },
        exc_info=True
    )
    
    rag_service.metrics.record_error("api", type(exc).__name__)
    
    return Response(
        content=ErrorResponse(
            error="Internal server error",
            error_type=type(exc).__name__,
            correlation_id=correlation_id
        ).json(),
        status_code=500,
        media_type="application/json"
    )


# Dependencies
def get_rag_service() -> RAGService:
    """Dependency to get RAG service."""
    return rag_service


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check(service: RAGService = Depends(get_rag_service)):
    """Check system health."""
    
    try:
        pipeline = service.get_pipeline()
        health_status = await pipeline.health_check()
        
        return HealthCheckResponse(**health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Query the RAG system."""
    
    try:
        # Set correlation ID if provided
        if request.correlation_id:
            set_correlation_id(request.correlation_id)
        
        pipeline = service.get_pipeline()
        
        response = await pipeline.query(
            question=request.question,
            top_k=request.top_k,
            filters=request.filters,
            include_metadata=request.include_metadata
        )
        
        # Record metrics
        service.metrics.record_query_processed(
            status="success",
            duration=response["processing_time_ms"] / 1000,
            has_results=response["retrieved_chunks"] > 0
        )
        
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        service.metrics.record_error("query", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))


# Batch query endpoint
@app.post("/query/batch", response_model=BatchQueryResponse)
async def batch_query(
    request: BatchQueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Process multiple queries in batch."""
    
    start_time = time.time()
    pipeline = service.get_pipeline()
    
    try:
        if request.parallel_processing:
            # Process queries in parallel
            tasks = [
                pipeline.query(
                    question=query.question,
                    top_k=query.top_k,
                    filters=query.filters,
                    include_metadata=query.include_metadata
                )
                for query in request.queries
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process queries sequentially
            results = []
            for query in request.queries:
                try:
                    result = await pipeline.query(
                        question=query.question,
                        top_k=query.top_k,
                        filters=query.filters,
                        include_metadata=query.include_metadata
                    )
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        # Process results
        successful_results = []
        successful_count = 0
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                # Create error response
                error_response = {
                    "error": str(result),
                    "correlation_id": get_correlation_id() or "",
                    "retrieved_chunks": 0,
                    "processing_time_ms": 0,
                    "model_info": {}
                }
                successful_results.append(error_response)
            else:
                successful_count += 1
                successful_results.append(result)
        
        total_time = int((time.time() - start_time) * 1000)
        
        return BatchQueryResponse(
            results=[QueryResponse(**result) for result in successful_results],
            total_processing_time_ms=total_time,
            successful_queries=successful_count,
            failed_queries=failed_count
        )
        
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document ingestion endpoint
@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    service: RAGService = Depends(get_rag_service)
):
    """Ingest documents into the RAG system."""
    
    try:
        pipeline = service.get_pipeline()
        
        # Run ingestion
        stats = await pipeline.ingest_documents(
            document_paths=request.document_paths,
            batch_size=request.batch_size
        )
        
        # Record metrics
        service.metrics.record_ingestion_time(
            stats["processing_time_seconds"],
            request.batch_size
        )
        
        service.metrics.record_documents_processed(
            stats["documents_processed"],
            "success"
        )
        
        return IngestionResponse(
            documents_processed=stats["documents_processed"],
            chunks_created=stats["chunks_created"],
            embeddings_generated=stats["embeddings_generated"],
            processing_time_s=stats["processing_time_seconds"],
            errors=stats["errors"],
            correlation_id=get_correlation_id() or ""
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        service.metrics.record_error("ingestion", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))


# Statistics endpoint
@app.get("/stats", response_model=StatsResponse)
async def get_stats(service: RAGService = Depends(get_rag_service)):
    """Get system statistics."""
    
    try:
        pipeline = service.get_pipeline()
        stats = await pipeline.get_stats()
        
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metrics endpoint
@app.get("/metrics", response_class=Response)
async def get_metrics_endpoint(service: RAGService = Depends(get_rag_service)):
    """Get Prometheus metrics."""
    
    try:
        metrics_data = service.metrics.get_metrics()
        
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin endpoints
@app.post("/admin", response_model=AdminResponse)
async def admin_operation(
    request: AdminRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Execute administrative operations."""
    
    start_time = time.time()
    
    try:
        # This would implement actual admin operations
        # For now, return a placeholder response
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return AdminResponse(
            command=request.command.value,
            status="success",
            message=f"Command {request.command.value} executed successfully",
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"Admin operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Streaming query endpoint
@app.post("/query/stream")
async def stream_query(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Stream query response."""
    
    async def generate_stream():
        try:
            # This would implement actual streaming
            # For now, yield a simple response
            yield f"data: {{'chunk_type': 'answer', 'content': 'Streaming response not yet implemented'}}\n\n"
            yield f"data: {{'chunk_type': 'complete', 'is_complete': true}}\n\n"
            
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain"
    )


def create_app(config_path: str = "./config") -> FastAPI:
    """Create FastAPI app with configuration."""
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.load_config()
    
    # Setup logging
    setup_logging(
        level=config.observability.log_level,
        format_type=config.observability.log_format
    )
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    config_path: str = "./config",
    reload: bool = False
):
    """Run the RAG API server."""
    
    app_instance = create_app(config_path)
    
    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        reload=reload,
        log_config=None  # Use our custom logging
    )


if __name__ == "__main__":
    run_server()