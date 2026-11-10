"""Failure modes the mock provider environment can produce."""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """A tool returned an error result (e.g. slot_taken, hold_expired)."""

    retryable = False

    def __init__(self, code: str, message: str = "", **details: Any) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message or code
        self.details = details


class ToolTimeout(ToolError):
    """The call timed out client-side. The side effect may or may not have happened."""

    retryable = True

    def __init__(self, message: str = "request timed out") -> None:
        super().__init__("timeout", message)


class TransientToolError(ToolError):
    """A transient server failure such as HTTP 503. Safe to retry with the same idempotency key."""

    retryable = True

    def __init__(self, code: str = "unavailable_503", message: str = "service unavailable") -> None:
        super().__init__(code, message)
