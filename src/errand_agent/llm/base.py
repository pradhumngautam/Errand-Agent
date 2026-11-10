"""Provider-neutral model interface.

The agent talks to `LLMClient` only. Implementations:
  * GeminiClient   - real model for the live demo
  * ScriptedLLM    - deterministic, offline, for evals
  * RecordingLLM / ReplayLLM - capture a real run and replay it offline
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON schema (object)


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict = Field(default_factory=dict)


class Message(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # role == "tool"
    tool_call_id: str | None = None
    name: str | None = None
    result: dict | None = None
    # Opaque provider payload, replayed verbatim (e.g. Gemini thought signatures).
    provider_raw: Any = Field(default=None, exclude=True)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    thoughts: str | None = None  # provider's thought summary, if any (shown in the trace, never re-sent)
    cost_usd: float | None = None  # real cost when the provider reports it (OpenRouter does)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = "unknown"
    provider_raw: Any = Field(default=None, exclude=True)


class LLMError(Exception):
    """The model provider failed (after the adapter's own retries)."""


# on_stream(kind, chunk): kind is "thought", "text", or "reset" (a retry restarted the stream)
StreamCallback = Callable[[str, str], None]


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 on_stream: StreamCallback | None = None) -> LLMResponse: ...


# USD per 1M tokens (input, output), list prices used for *estimated* cost.
# Free-tier calls cost nothing; the trace reports what the run would cost on paid tier.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3-pro-preview": (2.00, 12.00),
    "scripted": (0.0, 0.0),
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    price = PRICES.get(model) or next((v for k, v in PRICES.items() if model.startswith(k)), None)
    if price is None and model.startswith("gemini"):  # newer releases: assume the tier's latest list price
        price = (0.10, 0.40) if "lite" in model else (0.50, 3.00) if "flash" in model else (2.00, 12.00)
    if price is None:
        return None
    return round(usage.input_tokens / 1e6 * price[0] + usage.output_tokens / 1e6 * price[1], 6)


def estimate_tokens(text: str) -> int:
    """Conservative offline token estimate (~3.5 chars/token, rounded up).
    Used to enforce the context budget deterministically in offline evals;
    real usage from the provider is logged next to it."""
    return (len(text) * 2 + 6) // 7
