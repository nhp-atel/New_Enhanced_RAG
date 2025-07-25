"""Abstract interface for LLM clients."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator
from pydantic import BaseModel
from enum import Enum


class Role(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Chat message."""
    role: Role
    content: str


class LLMResponse(BaseModel):
    """Response from LLM."""
    content: str
    finish_reason: str
    token_usage: Optional[Dict[str, int]] = None
    model_name: str
    response_time_ms: int


class StreamingLLMResponse(BaseModel):
    """Streaming response chunk from LLM."""
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    is_complete: bool = False


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            LLMResponse containing the generated content
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> AsyncGenerator[StreamingLLMResponse, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Yields:
            StreamingLLMResponse chunks
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
            
        Raises:
            LLMError: If tokenization fails
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the LLM model."""
        pass
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Return the maximum context length for this model."""
        pass


class LLMError(Exception):
    """Exception raised when LLM operations fail."""
    pass