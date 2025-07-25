"""API request and response schemas."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class QueryRequest(BaseModel):
    """Request schema for querying the RAG system."""
    
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Number of relevant chunks to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for retrieval")
    include_metadata: bool = Field(default=True, description="Whether to include source metadata in response")
    correlation_id: Optional[str] = Field(default=None, description="Optional correlation ID for request tracing")


class SourceInfo(BaseModel):
    """Information about a source document."""
    
    chunk_id: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response schema for RAG queries."""
    
    answer: str
    correlation_id: str
    processing_time_ms: int
    retrieved_chunks: int
    model_info: Dict[str, str]
    sources: Optional[List[SourceInfo]] = None
    no_relevant_context: Optional[bool] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str
    error_type: str
    correlation_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class IngestionRequest(BaseModel):
    """Request schema for document ingestion."""
    
    document_paths: List[str] = Field(..., min_items=1, description="List of document paths to ingest")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size for processing")
    overwrite_existing: bool = Field(default=False, description="Whether to overwrite existing documents")


class IngestionResponse(BaseModel):
    """Response schema for document ingestion."""
    
    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    processing_time_s: float
    errors: List[str]
    correlation_id: str


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    components: Dict[str, str]
    timestamp: float


class StatsResponse(BaseModel):
    """System statistics response."""
    
    pipeline_config: Dict[str, Any]
    vector_store: Dict[str, Any]
    embedding_dimension: int


class ConfigUpdateRequest(BaseModel):
    """Request schema for updating configuration."""
    
    config_overrides: Dict[str, Any] = Field(..., description="Configuration values to override")


class ConfigResponse(BaseModel):
    """Configuration response schema."""
    
    current_config: Dict[str, Any]
    applied_overrides: Optional[Dict[str, Any]] = None


# Streaming response schemas
class StreamingQueryResponse(BaseModel):
    """Streaming query response chunk."""
    
    chunk_type: str = Field(..., pattern="^(context|answer|complete)$")
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_complete: bool = False


# Validation schemas
class DocumentValidationRequest(BaseModel):
    """Request to validate document format."""
    
    document_path: str
    check_encoding: bool = Field(default=True)
    check_size: bool = Field(default=True)


class DocumentValidationResponse(BaseModel):
    """Document validation response."""
    
    is_valid: bool
    file_size_bytes: int
    encoding: Optional[str] = None
    issues: List[str] = Field(default_factory=list)
    supported: bool


# Batch operation schemas
class BatchQueryRequest(BaseModel):
    """Request for batch query processing."""
    
    queries: List[QueryRequest] = Field(..., min_items=1, max_items=100)
    parallel_processing: bool = Field(default=True)


class BatchQueryResponse(BaseModel):
    """Response for batch queries."""
    
    results: List[QueryResponse]
    total_processing_time_ms: int
    successful_queries: int
    failed_queries: int


# Search and filter schemas
class SearchFilter(BaseModel):
    """Advanced search filter."""
    
    field: str
    operator: str = Field(..., pattern="^(eq|ne|gt|gte|lt|lte|in|contains)$")
    value: Any


class AdvancedQueryRequest(BaseModel):
    """Advanced query with multiple filters and options."""
    
    question: str = Field(..., min_length=1, max_length=2000)
    filters: List[SearchFilter] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank: bool = Field(default=False)
    explain_ranking: bool = Field(default=False)


# Admin schemas
class SystemCommand(str, Enum):
    """System administration commands."""
    RELOAD_CONFIG = "reload_config"
    CLEAR_CACHE = "clear_cache"
    REBUILD_INDEX = "rebuild_index"
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"


class AdminRequest(BaseModel):
    """Administrative operation request."""
    
    command: SystemCommand
    parameters: Optional[Dict[str, Any]] = None
    force: bool = Field(default=False)


class AdminResponse(BaseModel):
    """Administrative operation response."""
    
    command: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time_ms: int


# Monitoring schemas
class MetricsResponse(BaseModel):
    """Prometheus metrics response."""
    
    metrics: str
    format: str = "prometheus"
    timestamp: float


class LogsRequest(BaseModel):
    """Request for retrieving logs."""
    
    level: Optional[str] = Field(default=None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    since: Optional[str] = None  # ISO timestamp
    limit: int = Field(default=100, ge=1, le=10000)
    correlation_id: Optional[str] = None


class LogEntry(BaseModel):
    """Individual log entry."""
    
    timestamp: str
    level: str
    logger: str
    message: str
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LogsResponse(BaseModel):
    """Logs retrieval response."""
    
    logs: List[LogEntry] = Field(default_factory=list)
    total_entries: int
    has_more: bool