import json
import logging
from typing import Dict, Any, List, Optional, Generator
import requests

from equimind.config import settings
from equimind.providers.base import (
    LLMProvider,
    LLMMessage,
    ToolDefinition,
    ToolCall,
    TokenUsage,
    LLMResponse,
    Role,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Provider adapter for Anthropic Claude models (e.g. claude-3-5-sonnet, claude-3-haiku)."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        api_key = api_key or settings.anthropic_api_key
        base_url = base_url or settings.anthropic_base_url
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }

        system_prompt = ""
        anthropic_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt += f"{msg.content}\n"
            else:
                anthropic_messages.append({
                    "role": "user" if msg.role in (Role.USER, Role.TOOL) else "assistant",
                    "content": msg.content,
                })

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }

        if system_prompt.strip():
            payload["system"] = system_prompt.strip()

        if tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=settings.request_timeout_seconds
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_response(data)
        except Exception as e:
            logger.error(f"AnthropicProvider API call failed: {e}")
            raise RuntimeError(f"AnthropicProvider error: {e}") from e

    def generate_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        url = f"{self.base_url.rstrip('/')}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }

        system_prompt = ""
        anthropic_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt += f"{msg.content}\n"
            else:
                anthropic_messages.append({
                    "role": "user" if msg.role in (Role.USER, Role.TOOL) else "assistant",
                    "content": msg.content,
                })

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
            "stream": True,
        }

        if system_prompt.strip():
            payload["system"] = system_prompt.strip()

        resp = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=settings.request_timeout_seconds
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    try:
                        chunk_json = json.loads(line_str[6:].strip())
                        if chunk_json.get("type") == "content_block_delta":
                            delta = chunk_json.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                    except Exception:
                        continue

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        content_blocks = data.get("content", [])
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        usage = data.get("usage", {})
        token_usage = TokenUsage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        )

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
            role=Role.ASSISTANT,
            model_name=data.get("model", self.model_name),
            provider_name=self.provider_name,
            token_usage=token_usage,
            finish_reason=data.get("stop_reason"),
            raw_response=data,
        )
