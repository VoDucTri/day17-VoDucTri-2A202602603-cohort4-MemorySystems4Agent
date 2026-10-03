from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name, handling common aliases and misspellings."""
    if not value:
        return "gemini"
    val = value.strip().lower().replace("_", "-")
    alias_map = {
        "anthorpic": "anthropic",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "openai": "openai",
        "gpt": "openai",
        "custom": "custom",
        "openai-compatible": "custom",
        "google": "gemini",
        "google-genai": "gemini",
        "gemini": "gemini",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open-router": "openrouter",
    }
    if val in alias_map:
        return alias_map[val]
    raise ValueError(f"Unsupported provider: '{value}'. Supported: openai, custom, gemini, anthropic, ollama, openrouter")


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Supported providers:
    - openai -> ChatOpenAI
    - custom -> ChatOpenAI with base_url
    - gemini -> ChatGoogleGenerativeAI
    - anthropic -> ChatAnthropic
    - ollama -> ChatOllama
    - openrouter -> ChatOpenRouter
    """
    provider = normalize_provider(config.provider)
    temp = config.temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name or "gpt-4o-mini",
            temperature=temp,
            api_key=config.api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name or "gpt-3.5-turbo",
            temperature=temp,
            api_key=config.api_key or "custom-key",
            base_url=config.base_url or "http://localhost:8000/v1",
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name or "gemini-1.5-flash",
            temperature=temp,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name=config.model_name or "claude-3-5-sonnet-20241022",
            temperature=temp,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name or "llama3",
            temperature=temp,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
            return ChatOpenRouter(
                model=config.model_name or "meta-llama/llama-3.1-8b-instruct:free",
                temperature=temp,
                api_key=config.api_key,
            )
        except (ImportError, Exception):
            # Fallback to ChatOpenAI with OpenRouter base_url
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name or "meta-llama/llama-3.1-8b-instruct:free",
                temperature=temp,
                api_key=config.api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    raise ValueError(f"Unknown provider: {provider}")
