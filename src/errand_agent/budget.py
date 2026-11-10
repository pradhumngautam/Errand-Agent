"""Hard limits on model turns, tool calls and wall-clock time for one run."""

from __future__ import annotations

import time
from typing import Callable

from pydantic import BaseModel


class Limits(BaseModel):
    max_model_turns: int = 8
    max_tool_calls: int = 20
    max_wall_seconds: float = 90.0
    tool_retries: int = 2          # retries for transient tool errors
    book_retries: int = 2          # re-attempts of book() after a verified not-booked timeout
    status_check_retries: int = 3  # get_booking_status attempts when resolving ambiguity
    retry_backoff_seconds: float = 0.2
    # Calls that resolve an in-flight side effect (status checks) may exceed
    # max_tool_calls by this much. A limit must never stop us from finding out
    # whether we charged someone.
    essential_call_reserve: int = 6


class BudgetExceeded(Exception):
    def __init__(self, kind: str, limit: float) -> None:
        super().__init__(f"{kind} limit reached ({limit})")
        self.kind = kind
        self.limit = limit


class Budget:
    def __init__(self, limits: Limits | None = None, monotonic: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits or Limits()
        self._monotonic = monotonic
        self._start = monotonic()
        self.model_turns = 0
        self.tool_calls = 0

    @property
    def elapsed(self) -> float:
        return self._monotonic() - self._start

    def check_wall(self) -> None:
        if self.elapsed > self.limits.max_wall_seconds:
            raise BudgetExceeded("wall_clock_seconds", self.limits.max_wall_seconds)

    def charge_model_turn(self) -> None:
        self.check_wall()
        if self.model_turns >= self.limits.max_model_turns:
            raise BudgetExceeded("model_turns", self.limits.max_model_turns)
        self.model_turns += 1

    def charge_tool_call(self, essential: bool = False) -> None:
        cap = self.limits.max_tool_calls + (self.limits.essential_call_reserve if essential else 0)
        if not essential:
            self.check_wall()
        if self.tool_calls >= cap:
            raise BudgetExceeded("tool_calls", cap)
        self.tool_calls += 1

    def snapshot(self) -> dict:
        return {"model_turns": self.model_turns, "tool_calls": self.tool_calls, "elapsed_s": round(self.elapsed, 2)}
