import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


def _load_dotenv():
    """Loads environment variables from .env file if present in workspace."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_dotenv()


class EquiMindConfig(BaseModel):
    """Global configuration settings for EquiMind framework."""
    
    # Active default provider and fallback chain
    default_provider: str = Field(default_factory=lambda: os.getenv("EQUIMIND_DEFAULT_PROVIDER", "openai"))
    default_model: str = Field(default_factory=lambda: os.getenv("EQUIMIND_DEFAULT_MODEL", "gpt-4o"))
    fallback_providers: list[str] = Field(default_factory=lambda: ["openai", "anthropic", "gemini", "generic_openai"])
    
    # API Keys & Endpoints
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    anthropic_base_url: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"))
    
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_base_url: str = Field(default_factory=lambda: os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    
    deepseek_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    
    openrouter_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    openrouter_base_url: str = Field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    
    # Framework Limits & Settings
    max_context_tokens: int = Field(default_factory=lambda: int(os.getenv("EQUIMIND_MAX_CONTEXT_TOKENS", "32000")))
    request_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("EQUIMIND_REQUEST_TIMEOUT", "60")))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("EQUIMIND_MAX_RETRIES", "3")))


# Singleton instance
settings = EquiMindConfig()
