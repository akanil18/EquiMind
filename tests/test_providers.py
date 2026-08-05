import unittest
from equimind.providers.base import (
    Role,
    LLMMessage,
    ToolDefinition,
    ToolCall,
    LLMResponse,
)
from equimind.providers.mock_provider import MockProvider
from equimind.providers.openai_provider import OpenAIProvider
from equimind.providers.anthropic_provider import AnthropicProvider
from equimind.providers.gemini_provider import GeminiProvider
from equimind.providers.generic_openai_provider import GenericOpenAIProvider
from equimind.providers.factory import ProviderFactory


class TestProviders(unittest.TestCase):
    def test_mock_provider_generation(self):
        provider = MockProvider(model_name="mock-model", custom_response="Equity research test response")
        messages = [LLMMessage(role=Role.USER, content="Analyze NVDA")]
        
        response = provider.generate(messages)
        
        self.assertEqual(response.content, "Equity research test response")
        self.assertEqual(response.model_name, "mock-model")
        self.assertEqual(response.provider_name, "mock")
        self.assertEqual(response.token_usage.prompt_tokens, 10)
        self.assertEqual(response.token_usage.total_tokens, 30)

    def test_mock_provider_streaming(self):
        provider = MockProvider(custom_response="Hello world stream")
        messages = [LLMMessage(role=Role.USER, content="Test")]
        
        chunks = list(provider.generate_stream(messages))
        full_text = "".join(chunks).strip()
        
        self.assertEqual(full_text, "Hello world stream")

    def test_provider_factory_instantiation(self):
        openai_p = ProviderFactory.create("openai", model_name="gpt-4o")
        self.assertIsInstance(openai_p, OpenAIProvider)
        self.assertEqual(openai_p.model_name, "gpt-4o")
        self.assertEqual(openai_p.provider_name, "openai")

        anthropic_p = ProviderFactory.create("anthropic", model_name="claude-3-5-sonnet")
        self.assertIsInstance(anthropic_p, AnthropicProvider)
        self.assertEqual(anthropic_p.provider_name, "anthropic")

        gemini_p = ProviderFactory.create("gemini", model_name="gemini-1.5-pro")
        self.assertIsInstance(gemini_p, GeminiProvider)
        self.assertEqual(gemini_p.provider_name, "gemini")

        deepseek_p = ProviderFactory.create("deepseek", model_name="deepseek-chat")
        self.assertIsInstance(deepseek_p, GenericOpenAIProvider)
        self.assertEqual(deepseek_p.provider_name, "deepseek")

        ollama_p = ProviderFactory.create("ollama", model_name="llama3")
        self.assertIsInstance(ollama_p, GenericOpenAIProvider)
        self.assertEqual(ollama_p.provider_name, "ollama")

        mock_p = ProviderFactory.create("mock")
        self.assertIsInstance(mock_p, MockProvider)
        self.assertEqual(mock_p.provider_name, "mock")

    def test_factory_fallback_execution(self):
        messages = [LLMMessage(role=Role.USER, content="Should I buy AAPL?")]
        res = ProviderFactory.generate_with_fallback(messages=messages, preferred_provider="invalid_provider")
        
        self.assertIsInstance(res, LLMResponse)
        self.assertGreater(len(res.content), 0)


if __name__ == "__main__":
    unittest.main()
