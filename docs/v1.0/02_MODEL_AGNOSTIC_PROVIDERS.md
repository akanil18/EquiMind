# EquiMind v1.0: Model-Agnostic LLM Provider System (`equimind.providers`)

A core architectural pillar of EquiMind v1.0 is decoupling the reasoning engine from the orchestration framework.

---

## 🔌 Abstract Provider Interface

All model adapters inherit from `LLMProvider` in [equimind/providers/base.py](file:///home/anil-paliwal/Documents/Development/Quant_project/equimind/providers/base.py):

```python
class LLMProvider(ABC):
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, messages: List[LLMMessage], ...) -> LLMResponse:
        pass

    @abstractmethod
    def generate_stream(self, messages: List[LLMMessage], ...) -> Generator[str, None, None]:
        pass
```

---

## 🤖 Supported Model Adapters (v1.0)

| Provider Key | Adapter Class | Default Models | Features |
| :--- | :--- | :--- | :--- |
| `openai` | `OpenAIProvider` | `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini` | Tools, JSON Schema, Streaming |
| `anthropic` | `AnthropicProvider` | `claude-3-5-sonnet`, `claude-3-haiku` | Tools, System prompts, Streaming |
| `gemini` | `GeminiProvider` | `gemini-1.5-pro`, `gemini-2.5-flash` | System instructions, Function calls |
| `deepseek` | `GenericOpenAIProvider` | `deepseek-chat`, `deepseek-reasoner` | OpenAI-compatible format |
| `qwen` | `GenericOpenAIProvider` | `qwen-turbo`, `qwen-max` | OpenAI-compatible format |
| `ollama` | `GenericOpenAIProvider` | `llama3`, `mistral` | Local offline execution (`http://localhost:11434/v1`) |
| `openrouter` | `GenericOpenAIProvider` | Any OpenRouter model identifier | Unified API router |
| `mock` | `MockProvider` | `mock-gpt-4o` | Zero-cost deterministic offline testing |

---

## 🔄 ProviderFactory & Resilient Fallback Engine

The `ProviderFactory` in [equimind/providers/factory.py](file:///home/anil-paliwal/Documents/Development/Quant_project/equimind/providers/factory.py) manages provider registration, credential verification, and automated fallback execution:

```python
response = ProviderFactory.generate_with_fallback(
    messages=[LLMMessage(role=Role.USER, content="Analyze NVDA")],
    preferred_provider="deepseek",
)
```
