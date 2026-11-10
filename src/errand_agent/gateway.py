"""Single choke point for every tool call: budget, tracing, retries.

Nothing in the agent calls the mock service directly. This keeps the
"every tool call is traced and bounded" invariant true by construction.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from errand_agent.budget import Budget
from errand_agent.mocks.errors import ToolError, TransientToolError
from errand_agent.tracing import Tracer

TOOL_NAMES = (
    "search_providers", "get_provider", "hold_slot", "book",
    "get_booking_status", "cancel", "reschedule", "list_bookings",
)
SIDE_EFFECT_TOOLS = frozenset({"book", "cancel", "reschedule"})


class ToolGateway:
    def __init__(self, service: Any, tracer: Tracer, budget: Budget,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.service = service
        self.tracer = tracer
        self.budget = budget
        self.sleep = sleep

    def call(self, name: str, *, essential: bool = False, **args: Any) -> dict:
        if name not in TOOL_NAMES:
            raise ValueError(f"unknown tool {name}")
        self.budget.charge_tool_call(essential=essential)
        with self.tracer.timed() as t:
            try:
                result = getattr(self.service, name)(**args)
            except ToolError as exc:
                error = exc
            else:
                error = None
        if error is not None:
            self.tracer.emit("tool_call", tool=name, args=args, ok=False, error=error.code,
                             retryable=error.retryable, latency_ms=t["latency_ms"])
            raise error
        self.tracer.emit("tool_call", tool=name, args=args, ok=True, result=result, latency_ms=t["latency_ms"])
        return result

    def call_with_retry(self, name: str, *, retries: int, backoff: float, essential: bool = False,
                        **args: Any) -> dict:
        """Retry transient errors only. Only use this for idempotent or read-only calls."""
        attempt = 0
        while True:
            try:
                return self.call(name, essential=essential, **args)
            except ToolError as exc:
                if not isinstance(exc, TransientToolError) and exc.code != "timeout":
                    raise
                if attempt >= retries:
                    raise
                attempt += 1
                self.tracer.emit("retry", tool=name, attempt=attempt, error=exc.code)
                self.sleep(backoff * (2 ** (attempt - 1)))
