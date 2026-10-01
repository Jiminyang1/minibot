"""LLM profile resolution for MiniBot."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, cast


MaxOutputParameter = Literal["max_tokens", "max_completion_tokens"]


@dataclass(frozen=True)
class ModelCapabilities:
    """Hard limits advertised by one concrete provider/model pair."""

    context_window_tokens: int
    max_output_tokens: int
    max_input_tokens: int | None = None
    source: str = "catalog"


# Provider model-list endpoints usually expose identifiers, not token limits.
# Keep the small built-in catalog explicit and allow env overrides below for
# compatible gateways or newly released models.
_MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "deepseek-v4-pro": ModelCapabilities(1_048_576, 393_216),
    "deepseek-v4-flash": ModelCapabilities(1_048_576, 393_216),
    "gpt-5.4-mini": ModelCapabilities(
        400_000,
        128_000,
        max_input_tokens=272_000,
    ),
    "gpt-5.4": ModelCapabilities(1_050_000, 128_000),
    "gpt-5.4-pro": ModelCapabilities(1_050_000, 128_000),
}


@dataclass(frozen=True)
class OpenAICompatibleCompat:
    include_reasoning_content: bool = False
    supports_streaming: bool = True
    max_output_parameter: MaxOutputParameter = "max_completion_tokens"


@dataclass(frozen=True)
class LLMProfile:
    provider: str
    api: str
    model: str
    base_url: str | None
    api_key: str
    compat: OpenAICompatibleCompat
    capabilities: ModelCapabilities | None = None
    request_max_output_tokens: int | None = None


def build_llm_profile(
    *,
    model: str,
    context_window_tokens: int | None = None,
    model_max_input_tokens: int | None = None,
    model_max_output_tokens: int | None = None,
    request_max_output_tokens: int | None = None,
) -> LLMProfile:
    provider = _resolve_provider(model)
    base_url = _resolve_base_url(provider)
    api_key = _resolve_api_key(provider)
    capabilities = resolve_model_capabilities(
        model,
        context_window_tokens=context_window_tokens,
        model_max_input_tokens=model_max_input_tokens,
        model_max_output_tokens=model_max_output_tokens,
    )
    if request_max_output_tokens is not None:
        if request_max_output_tokens <= 0:
            raise ValueError("request_max_output_tokens 必须大于 0。")
        if request_max_output_tokens > capabilities.max_output_tokens:
            raise ValueError(
                "请求输出上限超过模型能力: "
                f"{request_max_output_tokens} > {capabilities.max_output_tokens}。"
            )
    compat = OpenAICompatibleCompat(
        include_reasoning_content=_should_send_reasoning_content(
            model,
            base_url or "",
            provider,
        ),
        supports_streaming=_should_stream(),
        max_output_parameter=_max_output_parameter(provider),
    )
    return LLMProfile(
        provider=provider,
        api="openai_chat_completions",
        model=model,
        base_url=base_url or None,
        api_key=api_key,
        compat=compat,
        capabilities=capabilities,
        request_max_output_tokens=request_max_output_tokens,
    )


def resolve_model_capabilities(
    model: str,
    *,
    context_window_tokens: int | None = None,
    model_max_input_tokens: int | None = None,
    model_max_output_tokens: int | None = None,
) -> ModelCapabilities:
    """Resolve hard model limits without guessing for unknown model IDs."""

    catalog = _catalog_capabilities(model)
    context_window = (
        context_window_tokens
        if context_window_tokens is not None
        else None if catalog is None else catalog.context_window_tokens
    )
    max_output = (
        model_max_output_tokens
        if model_max_output_tokens is not None
        else None if catalog is None else catalog.max_output_tokens
    )
    max_input = (
        model_max_input_tokens
        if model_max_input_tokens is not None
        else None if catalog is None else catalog.max_input_tokens
    )
    if context_window is None:
        raise ValueError(
            f"未知模型 {model!r} 的上下文窗口；请设置 "
            "MINIBOT_CONTEXT_WINDOW_TOKENS。"
        )
    if max_output is None:
        raise ValueError(
            f"未知模型 {model!r} 的最大输出；请设置 "
            "MINIBOT_MODEL_MAX_OUTPUT_TOKENS。"
        )
    if context_window <= 0:
        raise ValueError("context_window_tokens 必须大于 0。")
    if max_output <= 0:
        raise ValueError("model_max_output_tokens 必须大于 0。")
    if max_output >= context_window:
        raise ValueError("模型最大输出必须小于上下文窗口。")
    if max_input is not None:
        if max_input <= 0:
            raise ValueError("model_max_input_tokens 必须大于 0。")
        if max_input > context_window:
            raise ValueError("模型最大输入不能超过上下文窗口。")
    has_override = (
        context_window_tokens is not None
        or model_max_input_tokens is not None
        or model_max_output_tokens is not None
    )
    source = "env" if has_override else "catalog"
    return ModelCapabilities(
        context_window,
        max_output,
        max_input_tokens=max_input,
        source=source,
    )


def _catalog_capabilities(model: str) -> ModelCapabilities | None:
    normalized = model.strip().lower()
    exact = _MODEL_CAPABILITIES.get(normalized)
    if exact is not None:
        return exact
    for model_id, capabilities in _MODEL_CAPABILITIES.items():
        if normalized.startswith(f"{model_id}-20"):
            return capabilities
    return None


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
    provider_key = os.environ.get(env_name, "")
    if provider_key:
        return provider_key
    if env_name != "OPENAI_API_KEY":
        return os.environ.get("OPENAI_API_KEY", "")
    return ""


def _should_stream() -> bool:
    """Escape hatch for endpoints with broken SSE streaming."""
    raw = os.environ.get("MINIBOT_STREAMING", "auto").strip().lower()
    return raw not in {"0", "false", "no", "off", "never"}


def _max_output_parameter(provider: str) -> MaxOutputParameter:
    raw = os.environ.get("MINIBOT_MAX_OUTPUT_PARAMETER", "").strip().lower()
    if raw:
        if raw not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "MINIBOT_MAX_OUTPUT_PARAMETER 必须是 max_tokens 或 "
                "max_completion_tokens。"
            )
        return cast(MaxOutputParameter, raw)
    if provider == "deepseek":
        return "max_tokens"
    return "max_completion_tokens"


def _should_send_reasoning_content(model: str, base_url: str, provider: str) -> bool:
    raw = os.environ.get("MINIBOT_INCLUDE_REASONING_CONTENT", "auto").strip().lower()
    if raw in {"1", "true", "yes", "always"}:
        return True
    if raw in {"0", "false", "no", "never"}:
        return False

    if provider == "deepseek":
        return True
    return "deepseek" in base_url.lower() or model.lower().startswith("deepseek-")
