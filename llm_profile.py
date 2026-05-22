"""LLM profile resolution for MiniBot."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class OpenAICompatibleCompat:
    include_reasoning_content: bool = False


@dataclass(frozen=True)
class LLMProfile:
    provider: str
    api: str
    model: str
    base_url: str | None
    api_key: str
    compat: OpenAICompatibleCompat


def build_llm_profile(*, model: str) -> LLMProfile:
    provider = _resolve_provider(model)
    base_url = _resolve_base_url(provider)
    api_key = _resolve_api_key(provider)
    compat = OpenAICompatibleCompat(
        include_reasoning_content=_should_send_reasoning_content(
            model,
            base_url or "",
            provider,
        )
    )
    return LLMProfile(
        provider=provider,
        api="openai_chat_completions",
        model=model,
        base_url=base_url or None,
        api_key=api_key,
        compat=compat,
    )


def _resolve_provider(model: str) -> str:
    override = os.environ.get("MINIBOT_LLM_PROVIDER", "").strip().lower()
    if override:
        return override
    if model.lower().startswith("deepseek-"):
        return "deepseek"
    base_url = os.environ.get("OPENAI_BASE_URL", "").lower()
    if "deepseek" in base_url:
        return "deepseek"
    return "openai"


def _resolve_base_url(provider: str) -> str | None:
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url:
        return base_url
    if provider == "deepseek":
        return "https://api.deepseek.com"
    return None


def _resolve_api_key(provider: str) -> str:
    env_name = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }.get(provider, "OPENAI_API_KEY")
    return os.environ.get(env_name, "")


def _should_send_reasoning_content(model: str, base_url: str, provider: str) -> bool:
    raw = os.environ.get("MINIBOT_INCLUDE_REASONING_CONTENT", "auto").strip().lower()
    if raw in {"1", "true", "yes", "always"}:
        return True
    if raw in {"0", "false", "no", "never"}:
        return False

    if provider == "deepseek":
        return True
    return "deepseek" in base_url.lower() or model.lower().startswith("deepseek-")
