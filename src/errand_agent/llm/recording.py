"""Record a real model's responses to a cassette and replay them offline.

Replays return responses in order. The cassette also stores a hash of the
prompt for each step, so a replay that drifts from the recorded
conversation is reported instead of silently continuing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from errand_agent.llm.base import LLMClient, LLMError, LLMResponse, Message, StreamCallback, ToolSpec


def _digest(system: str, messages: list[Message]) -> str:
    payload = json.dumps([system] + [m.model_dump() for m in messages], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class RecordingLLM:
    def __init__(self, inner: LLMClient, path: str | Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.path = Path(path)
        self.entries: list[dict] = []

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 on_stream: StreamCallback | None = None) -> LLMResponse:
        resp = self.inner.complete(system, messages, tools, on_stream=on_stream)
        self.entries.append({"prompt_digest": _digest(system, messages), "response": resp.model_dump()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"model": self.model, "entries": self.entries}, indent=2))
        return resp


class ReplayLLM:
    def __init__(self, path: str | Path, strict: bool = False) -> None:
        data = json.loads(Path(path).read_text())
        self.model = data["model"]
        self.entries = data["entries"]
        self.strict = strict
        self.drift: list[int] = []
        self._i = 0

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 on_stream: StreamCallback | None = None) -> LLMResponse:
        if self._i >= len(self.entries):
            raise LLMError("replay cassette exhausted")
        entry = self.entries[self._i]
        if entry["prompt_digest"] != _digest(system, messages):
            self.drift.append(self._i)
            if self.strict:
                raise LLMError(f"replay drift at step {self._i}")
        self._i += 1
        resp = LLMResponse.model_validate(entry["response"])
        if on_stream:
            for kind, chunk in (("thought", resp.thoughts), ("text", resp.text)):
                if chunk:
                    on_stream(kind, chunk)
        return resp
