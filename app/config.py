"""Runtime settings, all read from environment variables (never from files in the repo).

Variable names (values are documented in README / .env.example, never committed):

    LLM_PROVIDER              openai (default) | anthropic | none
    LLM_MODEL                 model id; default gpt-4.1 (openai) / claude-opus-5 (anthropic)
    LLM_FALLBACK_MODEL        optional second model tried if the primary fails
    LLM_API_KEY               generic key; OPENAI_API_KEY / ANTHROPIC_API_KEY also work
    LLM_BASE_URL              optional OpenAI-compatible base URL (Groq, Gemini, OpenRouter, Ollama)
    LLM_EFFORT                anthropic only: low | medium | high | none   (default low)
    LLM_TIMEOUT_SECONDS       per model call (default 20)
    LLM_TOTAL_BUDGET_SECONDS  total interpretation budget per request (default 22, judge timeout is 30)
    LLM_CACHE_SIZE            interpretation cache entries (default 512)
    LOG_LEVEL                 default INFO
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_PROVIDER_ALIASES = {
    "openai": "openai",
    "openai_compatible": "openai",
    "openai-compatible": "openai",
    "groq": "openai",
    "gemini": "openai",
    "openrouter": "openai",
    "ollama": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "none": "none",
}

_DEFAULT_MODELS = {"openai": "gpt-4.1", "anthropic": "claude-opus-5", "none": ""}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_fallback_model: str | None
    llm_api_key: str | None
    llm_base_url: str | None
    llm_effort: str | None
    llm_timeout_s: float
    llm_total_budget_s: float
    llm_cache_size: int
    log_level: str

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def load_settings() -> Settings:
    raw_provider = (_env("LLM_PROVIDER", "openai") or "openai").lower()
    provider = _PROVIDER_ALIASES.get(raw_provider)
    if provider is None:
        raise ValueError(f"Unsupported LLM_PROVIDER '{raw_provider}' (use openai, anthropic or none)")

    if provider == "anthropic":
        api_key = _env("LLM_API_KEY") or _env("ANTHROPIC_API_KEY")
    else:
        api_key = _env("LLM_API_KEY") or _env("OPENAI_API_KEY")

    effort = (_env("LLM_EFFORT", "low") or "low").lower()
    if effort in ("none", "off", "0"):
        effort = None

    return Settings(
        llm_provider=provider,
        llm_model=_env("LLM_MODEL", _DEFAULT_MODELS[provider]) or _DEFAULT_MODELS[provider],
        llm_fallback_model=_env("LLM_FALLBACK_MODEL"),
        llm_api_key=api_key,
        llm_base_url=_env("LLM_BASE_URL"),
        llm_effort=effort,
        llm_timeout_s=_float("LLM_TIMEOUT_SECONDS", 20.0),
        llm_total_budget_s=_float("LLM_TOTAL_BUDGET_SECONDS", 22.0),
        llm_cache_size=int(_float("LLM_CACHE_SIZE", 512)),
        log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
    )
