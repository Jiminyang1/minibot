"""LLM client factory for MiniBot."""

from __future__ import annotations

from .llm import LLMClient
from .llm_profile import LLMProfile, build_llm_profile
from .llm_providers.openai_compatible import OpenAICompatibleClient


def build_llm_client(*, model: str) -> LLMClient:
    profile = build_llm_profile(model=model)
    return build_llm_client_from_profile(profile)


def build_llm_client_from_profile(profile: LLMProfile) -> LLMClient:
    if profile.api == "openai_chat_completions":
        return OpenAICompatibleClient(profile)
    raise NotImplementedError(f"不支持的 LLM api: {profile.api}")
