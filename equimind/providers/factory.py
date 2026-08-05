import logging
from typing import Dict, Type, Optional, List
from equimind.config import settings
from equimind.providers.base import LLMProvider, LLMMessage, LLMResponse, ToolDefinition
from equimind.providers.openai_provider import OpenAIProvider
from equimind.providers.anthropic_provider import AnthropicProvider
from equimind.providers.gemini_provider import GeminiProvider
from equimind.providers.generic_openai_provider import GenericOpenAIProvider
from equimind.providers.mock_provider import MockProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for instantiating and managing model-agnostic LLM providers."""

    _registry: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "generic_openai": GenericOpenAIProvider,
        "mock": MockProvider,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[LLMProvider]) -> None:
        """Register a new custom provider class."""
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def create(
        self,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> LLMProvider:
        """Instantiate a provider by name or fallback to default configuration."""
        name = (provider_name or settings.default_provider).lower()
        model = model_name or settings.default_model

        # Specific alias handling
        if name in ("deepseek", "qwen", "openrouter", "ollama"):
            if name == "ollama":
                base_url = base_url or settings.ollama_base_url
            elif name == "deepseek":
                base_url = base_url or settings.deepseek_base_url
                api_key = api_key or settings.deepseek_api_key
            elif name == "openrouter":
                base_url = base_url or settings.openrouter_base_url
                api_key = api_key or settings.openrouter_api_key

            return GenericOpenAIProvider(
                model_name=model,
                api_key=api_key,
                base_url=base_url,
                provider_label=name,
            )

        provider_cls = ProviderFactory._registry.get(name)
        if not provider_cls:
            logger.warning(f"Unknown provider '{name}'. Falling back to generic OpenAI format.")
            provider_cls = GenericOpenAIProvider

        return provider_cls(model_name=model, api_key=api_key, base_url=base_url)

    @classmethod
    def generate_with_fallback(
        cls,
        messages: List[LLMMessage],
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Executes LLM request trying preferred provider first, then looping through fallback chain."""
        candidates = [preferred_provider or settings.default_provider] + settings.fallback_providers
        unique_candidates = list(dict.fromkeys(candidates))

        last_exception = None
        for p_name in unique_candidates:
            try:
                provider = cls.create(provider_name=p_name, model_name=preferred_model)
                if not provider.is_available():
                    logger.debug(f"Provider {p_name} unavailable (missing credentials), skipping.")
                    continue
                return provider.generate(messages=messages, tools=tools, temperature=temperature)
            except Exception as e:
                logger.warning(f"Provider {p_name} failed: {e}. Trying next fallback.")
                last_exception = e

        # Final fallback to MockProvider if all live APIs fail or lack credentials
        logger.info("All configured live providers failed or unavailable. Using MockProvider.")
        mock = MockProvider(model_name="fallback-mock")
        return mock.generate(messages=messages, tools=tools, temperature=temperature)
