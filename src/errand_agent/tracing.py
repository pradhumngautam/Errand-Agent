"""Per-run JSONL traces: state transitions, model calls, tool calls, context, approvals."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
MAX_STR = 400


def sanitize(value: Any, max_str: int = MAX_STR) -> Any:
    """Redact PII-looking strings and truncate long free text before it is logged."""
    if isinstance(value, str):
        value = _PHONE.sub("[phone]", _EMAIL.sub("[email]", value))
        return value if len(value) <= max_str else value[:max_str] + f"…(+{len(value) - max_str} chars)"
    if isinstance(value, dict):
        return {k: sanitize(v, max_str) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v, max_str) for v in value]
    return value


class Tracer:
    """Collects events in memory (for tests and the UI) and appends them to a JSONL file."""

    def __init__(self, run_id: str, path: str | Path | None = None,
                 listener: Callable[[dict], None] | None = None, quiet: bool = False) -> None:
        self.run_id = run_id
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        self._seq = 0
        self._t0 = time.perf_counter()
        self.listeners: list[Callable[[dict], None]] = [listener] if listener else []
        # Quiet runs (e.g. a background poll that finds nothing) buffer in memory and are
        # only written/published if promote() is called.
        self.quiet = quiet
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _publish(self, event: dict) -> None:
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        for listener in self.listeners:
            listener(event)

    def promote(self) -> None:
        if self.quiet:
            self.quiet = False
            for event in self.events:
                self._publish(event)

    def emit(self, type_: str, **data: Any) -> dict:
        self._seq += 1
        event = {
            "run_id": self.run_id,
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_ms": round((time.perf_counter() - self._t0) * 1000, 1),
            "type": type_,
            **sanitize(data),
        }
        self.events.append(event)
        if not self.quiet:
            self._publish(event)
        return event

    @contextmanager
    def timed(self) -> Iterator[dict]:
        """Yields a dict; on exit it holds latency_ms for the wrapped block."""
        box: dict[str, float] = {}
        start = time.perf_counter()
        try:
            yield box
        finally:
            box["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)

    def of_type(self, type_: str) -> list[dict]:
        return [e for e in self.events if e["type"] == type_]
