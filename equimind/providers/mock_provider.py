import json
from typing import Dict, Any, List, Optional, Generator

from equimind.providers.base import (
    LLMProvider,
    LLMMessage,
    ToolDefinition,
    ToolCall,
    TokenUsage,
    LLMResponse,
    Role,
)


class MockProvider(LLMProvider):
    """Mock LLM Provider for unit testing, offline execution, and fallback validation."""

    def __init__(
        self,
        model_name: str = "mock-gpt-4o",
        api_key: Optional[str] = "mock-key",
        base_url: Optional[str] = "http://mock-url",
        custom_response: Optional[str] = None,
        custom_tool_calls: Optional[List[ToolCall]] = None,
    ):
        super().__init__(model_name=model_name, api_key=api_key or "mock-key", base_url=base_url or "http://mock-url")
        self.custom_response = custom_response
        self.custom_tool_calls = custom_tool_calls

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.role == Role.USER:
                last_user_msg = msg.content
                break

        response_content = self.custom_response or f"[Mock Response for: '{last_user_msg}']"

        return LLMResponse(
            content=response_content,
            tool_calls=self.custom_tool_calls,
            role=Role.ASSISTANT,
            model_name=self.model_name,
            provider_name=self.provider_name,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            finish_reason="stop",
            raw_response={"mock": True},
        )

    def generate_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        res = self.custom_response or "Mock streamed response token by token."
        words = res.split(" ")
        for word in words:
            yield word + " "
