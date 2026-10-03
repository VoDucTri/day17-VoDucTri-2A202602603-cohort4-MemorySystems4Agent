from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


import os
from dotenv import load_dotenv


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load environment variables from .env if present
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    data_dir = (root / "data").resolve()
    state_dir = (root / "state").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Determine provider and model
    provider_raw = os.getenv("LLM_PROVIDER", "gemini")
    provider = normalize_provider(provider_raw)

    api_key = None
    base_url = None
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model_name = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("LLM_MODEL", "llama3")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model_name = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
        model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    else:
        model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        api_key=api_key,
        base_url=base_url,
    )

    judge_config = ProviderConfig(
        provider=provider,
        model_name=os.getenv("JUDGE_MODEL", model_name),
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config,
    )
