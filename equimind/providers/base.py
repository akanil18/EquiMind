from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Generator
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolDefinition(BaseModel):
    """Specification for a tool executable by LLM agents."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema specification


class ToolCall(BaseModel):
    """Container for tool calls requested by an LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


class LLMMessage(BaseModel):
    """Standardized representation of a conversation message."""
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class TokenUsage(BaseModel):
    """Token usage metadata."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Unified response object returned by all LLM Providers."""
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    role: Role = Role.ASSISTANT
    model_name: str
    provider_name: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract Base Class for all LLM Providers in EquiMind."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the canonical name of the provider."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Generates a synchronous response from the model."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Streams text chunks from the model response."""
        pass

    def is_available(self) -> bool:
        """Checks if the provider has necessary API credentials configured."""
        return bool(self.api_key) or self.provider_name in ("ollama", "mock")
