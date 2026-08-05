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


class OpenAIProvider(LLMProvider):
    """Provider adapter for OpenAI models (e.g. gpt-4o, gpt-4o-mini, o1, o3-mini)."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        api_key = api_key or settings.openai_api_key
        base_url = base_url or settings.openai_base_url
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or ''}",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [self._format_message(m) for m in messages],
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if response_format:
            payload["response_format"] = response_format

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
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
            logger.error(f"OpenAIProvider API call failed: {e}")
            raise RuntimeError(f"OpenAIProvider error: {e}") from e

    def generate_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or ''}",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [self._format_message(m) for m in messages],
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        resp = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=settings.request_timeout_seconds
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_content = line_str[6:].strip()
                    if data_content == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_content)
                        delta = chunk_json["choices"][0]["delta"]
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                    except Exception:
                        continue

    def _format_message(self, msg: LLMMessage) -> Dict[str, Any]:
        formatted: Dict[str, Any] = {"role": msg.role.value, "content": msg.content}
        if msg.name:
            formatted["name"] = msg.name
        if msg.tool_call_id:
            formatted["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            formatted["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        return formatted

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        choice = data["choices"][0]
        message_data = choice["message"]
        content = message_data.get("content") or ""

        tool_calls = None
        if "tool_calls" in message_data and message_data["tool_calls"]:
            tool_calls = []
            for raw_tc in message_data["tool_calls"]:
                try:
                    args = json.loads(raw_tc["function"]["arguments"])
                except Exception:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=raw_tc.get("id", ""),
                        name=raw_tc["function"]["name"],
                        arguments=args,
                    )
                )

        usage_data = data.get("usage", {})
        token_usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            role=Role.ASSISTANT,
            model_name=data.get("model", self.model_name),
            provider_name=self.provider_name,
            token_usage=token_usage,
            finish_reason=choice.get("finish_reason"),
            raw_response=data,
        )
