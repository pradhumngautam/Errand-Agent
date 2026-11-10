"""Gemini adapter (google-genai SDK) for the live demo."""

from __future__ import annotations

import os
import time
import uuid

from errand_agent.llm.base import LLMError, LLMResponse, Message, StreamCallback, ToolCall, ToolSpec, Usage

_SKIP = ("image", "tts", "live", "audio", "embedding", "omni", "preview", "research")


def _version_key(name: str) -> tuple:
    nums = [int(n) for n in __import__("re").findall(r"\d+", name)]
    return tuple(nums)


class GeminiClient:
    """Picks the newest flash model that actually answers (listed models can be
    retired for new keys or overloaded), and fails over to the next one."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 temperature: float = 0.2, max_retries: int = 3, cache_path: str | None = None) -> None:
        from google import genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        from google.genai import types

        # Our own loop owns retries and model failover; the SDK's hidden retries made a
        # single overloaded model stall startup for minutes.
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)))
        self._probe_config = types.GenerateContentConfig(http_options=types.HttpOptions(timeout=15_000))
        self.temperature = temperature
        self.max_retries = max_retries
        self.failovers: list[dict] = []
        pinned = model or os.environ.get("GEMINI_MODEL")
        self.candidates = [pinned] if pinned else self._candidates()
        self.cache_path = cache_path
        cached = self._read_cache()
        if cached in self.candidates:  # try last session's working model first
            self.candidates.remove(cached)
            self.candidates.insert(0, cached)
        self.model = self._probe()
        self._write_cache()

    def _read_cache(self) -> str | None:
        try:
            return open(self.cache_path).read().strip() if self.cache_path else None
        except OSError:
            return None

    def _write_cache(self) -> None:
        if self.cache_path:
            try:
                with open(self.cache_path, "w") as fh:
                    fh.write(self.model)
            except OSError:
                pass

    def _candidates(self) -> list[str]:
        try:
            models = list(self._client.models.list())
        except Exception as exc:  # noqa: BLE001 - surface a clear setup error
            raise LLMError(f"could not list Gemini models: {exc}") from exc
        names = [m.name.removeprefix("models/") for m in models
                 if "generateContent" in (m.supported_actions or [])]
        flash = [n for n in names if "flash" in n and not any(t in n for t in _SKIP)]
        # full flash models newest-first, then flash-lite as fallbacks (separate free-tier quotas)
        return sorted(flash, key=lambda n: ("lite" in n, tuple(-x for x in _version_key(n))))

    def _probe(self) -> str:
        errors = []
        for name in list(self.candidates):
            try:
                self._client.models.generate_content(model=name, contents="ping", config=self._probe_config)
                self.candidates = self.candidates[self.candidates.index(name):]
                return name
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {str(exc)[:60]}")
        raise LLMError("no usable Gemini model: " + "; ".join(errors))

    # ----------------------------------------------------------- conversion
    def _contents(self, messages: list[Message]) -> list:
        from google.genai import types

        contents: list = []
        pending_results: list = []

        def flush() -> None:
            if pending_results:
                contents.append(types.Content(role="user", parts=list(pending_results)))
                pending_results.clear()

        for m in messages:
            if m.role == "tool":
                pending_results.append(types.Part.from_function_response(
                    name=m.name or "tool", response=m.result or {}))
                continue
            flush()
            if m.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m.content or "")]))
            elif m.provider_raw is not None:
                contents.append(m.provider_raw)  # keeps thought signatures intact
            else:
                parts = [types.Part(text=m.content)] if m.content else []
                parts += [types.Part(function_call=types.FunctionCall(name=c.name, args=c.args)) for c in m.tool_calls]
                contents.append(types.Content(role="model", parts=parts or [types.Part(text="")]))
        flush()
        return contents

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 on_stream: StreamCallback | None = None) -> LLMResponse:
        """One model call. With `on_stream`, thought summaries and reply text are pushed
        as they arrive ("thought" / "text" chunks; "reset" if a retry restarts the stream)."""
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                for t in tools])] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )
        contents = self._contents(messages)
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                parts, um = self._call(contents, config, on_stream)
                break
            except errors.APIError as exc:
                if on_stream:
                    on_stream("reset", "")
                code = getattr(exc, "code", None)
                daily_quota = code == 429 and "PerDay" in str(exc)
                if len(self.candidates) > 1 and (code == 404 or daily_quota or (code in (500, 503, 504) and attempt >= 1)):
                    # model retired / out of daily quota / overloaded: fail over to the next one
                    # (history without the old model's thought signatures still works)
                    self.failovers.append({"from": self.model, "code": code, "to": self.candidates[1]})
                    self.candidates = self.candidates[1:]
                    self.model = self.candidates[0]
                    self._write_cache()
                    contents = self._contents([m.model_copy(update={"provider_raw": None}) for m in messages])
                    continue
                retryable = code in (429, 500, 502, 503, 504)
                if not retryable or attempt == self.max_retries:
                    raise LLMError(f"gemini error {code or '?'}: {str(exc)[:300]}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 20)
            except Exception as exc:  # network etc.
                if on_stream:
                    on_stream("reset", "")
                if attempt == self.max_retries:
                    raise LLMError(f"gemini request failed: {exc}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 20)

        text = "".join(p.text for p in parts if p.text and not p.thought) or None
        thoughts = "".join(p.text for p in parts if p.text and p.thought) or None
        tool_calls = [ToolCall(id=p.function_call.id or f"call_{uuid.uuid4().hex[:8]}", name=p.function_call.name,
                               args=dict(p.function_call.args or {}))
                      for p in parts if p.function_call]
        usage = Usage(input_tokens=(um.prompt_token_count or 0) if um else 0,
                      output_tokens=((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0)
        content = types.Content(role="model", parts=parts) if parts else None
        return LLMResponse(text=text, thoughts=thoughts, tool_calls=tool_calls, usage=usage, model=self.model,
                           provider_raw=content)

    def _call(self, contents: list, config, on_stream: StreamCallback | None) -> tuple[list, object]:
        if on_stream is None:
            resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
            cand = (resp.candidates or [None])[0]
            content = getattr(cand, "content", None)
            return list(content.parts if content and content.parts else []), resp.usage_metadata
        parts: list = []
        usage = None
        for chunk in self._client.models.generate_content_stream(model=self.model, contents=contents, config=config):
            usage = chunk.usage_metadata or usage
            cand = (chunk.candidates or [None])[0]
            content = getattr(cand, "content", None)
            for p in (content.parts if content and content.parts else []):
                parts.append(p)  # kept verbatim: thought signatures ride on these parts
                if p.text:
                    on_stream("thought" if p.thought else "text", p.text)
        return parts, usage
