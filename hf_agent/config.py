from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    provider: str  # "openai" | "gemini"
    model: str

    @property
    def label(self) -> str:
        return f"[{self.provider}] {self.model}"


def _split_models(value: str | None) -> list[str]:
    if not value:
        return []
    return [m.strip() for m in value.split(",") if m.strip()]


def available_models() -> list[ModelChoice]:
    """Read configured models from environment.

    Each provider contributes models only if its API key is set. Both a single
    `*_MODEL` and a comma-separated `*_MODELS` are accepted; `*_MODELS` wins.
    """
    choices: list[ModelChoice] = []
    if os.getenv("OPENAI_API_KEY"):
        for m in _split_models(os.getenv("OPENAI_MODELS") or os.getenv("OPENAI_MODEL")):
            choices.append(ModelChoice("openai", m))
    if os.getenv("GEMINI_API_KEY"):
        for m in _split_models(os.getenv("GEMINI_MODELS") or os.getenv("GEMINI_MODEL")):
            choices.append(ModelChoice("gemini", m))
    return choices
