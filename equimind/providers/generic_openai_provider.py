from typing import Optional
from equimind.config import settings
from equimind.providers.openai_provider import OpenAIProvider


class GenericOpenAIProvider(OpenAIProvider):
    """Generic OpenAI-compatible provider for DeepSeek, Qwen, Ollama, OpenRouter, vLLM, LM Studio."""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_label: str = "generic_openai",
    ):
        base_url = base_url or settings.deepseek_base_url
        api_key = api_key or settings.deepseek_api_key or "not-needed"
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)
        self._provider_label = provider_label

    @property
    def provider_name(self) -> str:
        return self._provider_label
