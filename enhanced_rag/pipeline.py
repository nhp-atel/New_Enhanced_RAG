"""Main RAG pipeline orchestrator."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import uuid

from .interfaces import Embedder, VectorStore, LLMClient, DocumentLoader
from .interfaces.vector_store import Document, SearchResult
from .interfaces.llm_client import Message, Role
from .core.chunking import ChunkerFactory, ChunkingStrategy, TextChunk
from .utils.config import RAGConfig
from .utils.retry import retry_on_network_error

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""
    
    def __init__(
        self,
        config: RAGConfig,
        embedder: Embedder,
        vector_store: VectorStore,
        llm_client: LLMClient,
        document_loaders: Optional[Dict[str, DocumentLoader]] = None
    ):
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.document_loaders = document_loaders or {}
        
        # Create text chunker
        self.chunker = ChunkerFactory.create_chunker(
            strategy=ChunkingStrategy(config.chunking.strategy),
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap
        )
        
        # Load prompt templates
        self._load_prompt_templates()
        
        logger.info("RAG pipeline initialized")
    
    def _load_prompt_templates(self):
        """Load prompt templates from configuration."""
        # This would load from prompts.yaml in a real implementation
        self.prompts = {
            "system": "You are a helpful AI assistant that provides accurate responses based on context.",
            "default": "Based on the following context, please answer the question:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            "no_context": "I don't have enough relevant information to answer your question about '{query}'."
        }
    
    async def ingest_documents(
        self, 
        document_paths: List[str],
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            document_paths: List of document paths to ingest
            batch_size: Batch size for processing
            
        Returns:
            Ingestion statistics
        """
        start_time = time.time()
        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": []
        }
        
        logger.info(f"Starting ingestion of {len(document_paths)} documents")
        
        # Process documents in batches
        for i in range(0, len(document_paths), batch_size):
            batch_paths = document_paths[i:i + batch_size]
            
            try:
                await self._process_document_batch(batch_paths, stats)
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                stats["errors"].append(f"Batch {i//batch_size + 1}: {str(e)}")
        
        # Save vector store index
        await self.vector_store.save_index(self.config.vector_store.persistence_path)
        
        processing_time = time.time() - start_time
        stats["processing_time_seconds"] = processing_time
        
        logger.info(f"Ingestion completed in {processing_time:.2f}s: {stats}")
        return stats
    
    async def _process_document_batch(
        self, 
        document_paths: List[str], 
        stats: Dict[str, Any]
    ):
        """Process a batch of documents."""
        
        # Load documents
        loaded_docs = []
        for path in document_paths:
            try:
                doc = await self._load_document(path)
                if doc:
                    loaded_docs.append(doc)
                    stats["documents_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to load document {path}: {e}")
                stats["errors"].append(f"Load {path}: {str(e)}")
        
        if not loaded_docs:
            return
        
        # Chunk documents
        all_chunks = []
        documents_for_storage = []
        
        for doc in loaded_docs:
            try:
                chunks = self.chunker.chunk_text(
                    doc.content,
                    metadata={
                        "source": doc.source,
                        "document_id": doc.document_id,
                        **doc.metadata
                    }
                )
                
                for chunk in chunks:
                    # Create document for vector storage
                    vector_doc = Document(
                        id=chunk.chunk_id,
                        content=chunk.content,
                        metadata=chunk.metadata
                    )
                    
                    all_chunks.append(chunk.content)
                    documents_for_storage.append(vector_doc)
                
                stats["chunks_created"] += len(chunks)
                
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.document_id}: {e}")
                stats["errors"].append(f"Chunk {doc.document_id}: {str(e)}")
        
        if not all_chunks:
            return
        
        # Generate embeddings
        try:
            embedding_result = await self._generate_embeddings_with_retry(all_chunks)
            embeddings = embedding_result.embeddings
            stats["embeddings_generated"] += len(embeddings)
            
            # Store in vector database
            await self.vector_store.add_documents(documents_for_storage, embeddings)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings or store documents: {e}")
            stats["errors"].append(f"Embeddings/Storage: {str(e)}")
    
    async def _load_document(self, path: str):
        """Load a document using appropriate loader."""
        path_obj = Path(path)
        
        # Find appropriate loader
        for extension, loader in self.document_loaders.items():
            if path_obj.suffix.lower() == extension.lower():
                return await loader.load_document(path)
        
        # Default text loader if no specific loader found
        if ".txt" in self.document_loaders:
            return await self.document_loaders[".txt"].load_document(path)
        
        raise ValueError(f"No loader found for file: {path}")
    
    @retry_on_network_error(max_attempts=3)
    async def _generate_embeddings_with_retry(self, texts: List[str]):
        """Generate embeddings with retry logic."""
        return await self.embedder.embed_texts(
            texts, 
            batch_size=self.config.embedding.batch_size
        )
    
    async def query(
        self, 
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system for an answer.
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            filters: Optional metadata filters
            include_metadata: Whether to include source metadata
            
        Returns:
            Query response with answer and metadata
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Processing query: {question[:100]}...", extra={"correlation_id": correlation_id})
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding_with_retry(question)
            
            # Retrieve relevant chunks
            top_k = top_k or self.config.retrieval.top_k
            search_results = await self.vector_store.similarity_search(
                query_embedding, 
                top_k=top_k,
                filters=filters
            )
            
            if not search_results:
                return self._create_no_context_response(question, correlation_id)
            
            # Filter by score threshold
            filtered_results = [
                result for result in search_results 
                if result.score >= self.config.retrieval.score_threshold
            ]
            
            if not filtered_results:
                return self._create_no_context_response(question, correlation_id)
            
            # Generate answer
            answer = await self._generate_answer_with_retry(question, filtered_results)
            
            # Prepare response
            response = {
                "answer": answer.content,
                "correlation_id": correlation_id,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "retrieved_chunks": len(filtered_results),
                "model_info": {
                    "embedder": self.embedder.model_name,
                    "llm": self.llm_client.model_name
                }
            }
            
            if include_metadata:
                response["sources"] = [
                    {
                        "chunk_id": result.document.id,
                        "content": result.document.content[:200] + "...",
                        "score": result.score,
                        "metadata": result.document.metadata
                    }
                    for result in filtered_results
                ]
            
            logger.info(f"Query completed successfully", extra={"correlation_id": correlation_id})
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}", extra={"correlation_id": correlation_id})
            return {
                "error": str(e),
                "correlation_id": correlation_id,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
    
    @retry_on_network_error(max_attempts=3)
    async def _generate_query_embedding_with_retry(self, question: str):
        """Generate query embedding with retry logic."""
        return await self.embedder.embed_query(question)
    
    @retry_on_network_error(max_attempts=3)
    async def _generate_answer_with_retry(
        self, 
        question: str, 
        search_results: List[SearchResult]
    ):
        """Generate answer with retry logic."""
        # Prepare context
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[{i}] {result.document.content}")
        
        context = "\n\n".join(context_parts)
        
        # Prepare prompt
        prompt = self.prompts["default"].format(
            context=context,
            question=question
        )
        
        # Generate response
        messages = [
            Message(role=Role.SYSTEM, content=self.prompts["system"]),
            Message(role=Role.USER, content=prompt)
        ]
        
        return await self.llm_client.generate_response(
            messages,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature
        )
    
    def _create_no_context_response(self, question: str, correlation_id: str) -> Dict[str, Any]:
        """Create response when no relevant context is found."""
        return {
            "answer": self.prompts["no_context"].format(query=question),
            "correlation_id": correlation_id,
            "retrieved_chunks": 0,
            "no_relevant_context": True
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "pipeline_config": {
                "chunk_size": self.config.chunking.chunk_size,
                "overlap": self.config.chunking.overlap,
                "top_k": self.config.retrieval.top_k,
                "embedding_model": self.embedder.model_name,
                "llm_model": self.llm_client.model_name
            },
            "vector_store": vector_stats,
            "embedding_dimension": self.embedder.embedding_dimension
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        # Check embedder
        try:
            await self.embedder.embed_query("test")
            health["components"]["embedder"] = "healthy"
        except Exception as e:
            health["components"]["embedder"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check vector store
        try:
            stats = self.vector_store.get_stats()
            health["components"]["vector_store"] = "healthy"
        except Exception as e:
            health["components"]["vector_store"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check LLM client
        try:
            test_messages = [Message(role=Role.USER, content="Hello")]
            await self.llm_client.generate_response(test_messages, max_tokens=10)
            health["components"]["llm_client"] = "healthy"
        except Exception as e:
            health["components"]["llm_client"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        return health