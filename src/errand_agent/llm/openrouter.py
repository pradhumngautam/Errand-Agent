"""OpenRouter adapter (OpenAI-compatible chat completions over HTTP, streamed).

Used for the live demo when OPENROUTER_API_KEY is set. It reports the real
per-call cost OpenRouter returns, and preserves `reasoning_details` across
tool turns (needed by reasoning models such as Gemini 3 for thought
signatures).
"""

from __future__ import annotations

import json
import os
import time
import uuid

import httpx

from errand_agent.llm.base import LLMError, LLMResponse, Message, StreamCallback, ToolCall, ToolSpec, Usage

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.6-flash"
FALLBACK_MODELS = ["anthropic/claude-haiku-5.5", "deepseek/deepseek-v4.1-flash"]


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, temperature: float = 0.2,
                 max_retries: int = 3, timeout: float = 90.0) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self.model = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_cost_usd: float | None = None
        self.failovers: list[dict] = []
        self.fallbacks = [m for m in FALLBACK_MODELS if m != self.model]

    # ----------------------------------------------------------- conversion
    @staticmethod
    def _messages(system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content or ""})
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id or "", "content": json.dumps(m.result or {})})
            else:
                msg: dict = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                                         for c in m.tool_calls]
                if isinstance(m.provider_raw, dict) and m.provider_raw.get("reasoning_details"):
                    msg["reasoning_details"] = m.provider_raw["reasoning_details"]
                out.append(msg)
        return out

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 on_stream: StreamCallback | None = None) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "temperature": self.temperature,
            # OpenRouter reserves credit for max_tokens up front; uncapped (65k) is refused on small balances.
            "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "2048")),
            "stream": True,
            "usage": {"include": True},
            "reasoning": {"effort": "low"},
        }
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                               "parameters": t.parameters}} for t in tools]
        delay = 1.5
        for attempt in range(self.max_retries + 1):
            try:
                return self._stream(body, on_stream)
            except _Retryable as exc:
                if on_stream:
                    on_stream("reset", "")
                if attempt == self.max_retries:
                    if self.fallbacks and exc.status in (404, 429, 503, 529):
                        nxt = self.fallbacks.pop(0)
                        self.failovers.append({"from": self.model, "code": exc.status, "to": nxt})
                        self.model = body["model"] = nxt
                        continue
                    raise LLMError(f"openrouter {exc.status}: {exc}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 15)
        raise LLMError("openrouter: retries exhausted")

    def _stream(self, body: dict, on_stream: StreamCallback | None) -> LLMResponse:
        text, thoughts = "", ""
        calls: dict[int, dict] = {}
        reasoning_details: list = []
        usage = Usage()
        cost = None
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": "Errand Agent (local demo)"}
        try:
            with httpx.stream("POST", URL, json=body, headers=headers, timeout=self.timeout) as r:
                if r.status_code != 200:
                    detail = r.read().decode(errors="replace")[:300]
                    if r.status_code in (408, 429, 500, 502, 503, 504, 529):
                        raise _Retryable(r.status_code, detail)
                    raise LLMError(f"openrouter {r.status_code}: {detail}")
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        err = chunk["error"]
                        raise _Retryable(int(err.get("code") or 500), str(err.get("message"))[:300])
                    if chunk.get("usage"):
                        u = chunk["usage"]
                        usage = Usage(input_tokens=u.get("prompt_tokens") or 0,
                                      output_tokens=u.get("completion_tokens") or 0)
                        cost = u.get("cost")
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning"):
                            thoughts += delta["reasoning"]
                            if on_stream:
                                on_stream("thought", delta["reasoning"])
                        if delta.get("reasoning_details"):
                            reasoning_details.extend(delta["reasoning_details"])
                        if delta.get("content"):
                            text += delta["content"]
                            if on_stream:
                                on_stream("text", delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["args"] += fn.get("arguments") or ""
        except httpx.HTTPError as exc:
            raise _Retryable(599, f"network: {exc}") from exc

        tool_calls = []
        for i in sorted(calls):
            c = calls[i]
            try:
                args = json.loads(c["args"]) if c["args"].strip() else {}
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": c["args"][:500]}  # the tool layer returns bad_arguments
            tool_calls.append(ToolCall(id=c["id"] or f"call_{uuid.uuid4().hex[:8]}", name=c["name"], args=args))
        self.last_cost_usd = cost
        return LLMResponse(text=text or None, thoughts=thoughts or None, tool_calls=tool_calls, usage=usage,
                           model=self.model, provider_raw={"reasoning_details": reasoning_details},
                           cost_usd=cost)


class _Retryable(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
