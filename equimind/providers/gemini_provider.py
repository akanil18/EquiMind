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


class GeminiProvider(LLMProvider):
    """Provider adapter for Google Gemini models (e.g. gemini-1.5-pro, gemini-2.5-flash)."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        api_key = api_key or settings.gemini_api_key
        base_url = base_url or settings.gemini_base_url
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/models/{self.model_name}:generateContent?key={self.api_key or ''}"
        headers = {"Content-Type": "application/json"}

        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                role_str = "user" if msg.role in (Role.USER, Role.TOOL) else "model"
                contents.append({
                    "role": role_str,
                    "parts": [{"text": msg.content}],
                })

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ]

        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=settings.request_timeout_seconds
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_response(data)
        except Exception as e:
            logger.error(f"GeminiProvider API call failed: {e}")
            raise RuntimeError(f"GeminiProvider error: {e}") from e

    def generate_stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        url = f"{self.base_url.rstrip('/')}/models/{self.model_name}:streamGenerateContent?key={self.api_key or ''}"
        headers = {"Content-Type": "application/json"}

        contents = []
        for msg in messages:
            role_str = "user" if msg.role in (Role.USER, Role.TOOL, Role.SYSTEM) else "model"
            contents.append({
                "role": role_str,
                "parts": [{"text": msg.content}],
            })

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }

        resp = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=settings.request_timeout_seconds
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                try:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        line_str = line_str[6:]
                    chunk_data = json.loads(line_str)
                    candidates = chunk_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for p in parts:
                            if "text" in p:
                                yield p["text"]
                except Exception:
                    continue

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        candidates = data.get("candidates", [])
        if not candidates:
            return LLMResponse(
                content="",
                role=Role.ASSISTANT,
                model_name=self.model_name,
                provider_name=self.provider_name,
                raw_response=data,
            )

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        text_content = ""
        tool_calls = []

        for p in parts:
            if "text" in p:
                text_content += p["text"]
            elif "functionCall" in p:
                fc = p["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=fc.get("name", ""),
                        name=fc.get("name", ""),
                        arguments=fc.get("args", {}),
                    )
                )

        usage_meta = data.get("usageMetadata", {})
        token_usage = TokenUsage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
            total_tokens=usage_meta.get("totalTokenCount", 0),
        )

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
            role=Role.ASSISTANT,
            model_name=self.model_name,
            provider_name=self.provider_name,
            token_usage=token_usage,
            finish_reason=candidate.get("finishReason"),
            raw_response=data,
        )
