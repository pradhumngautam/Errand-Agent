"""Injectable clock so hold expiry, "this week" and restarts are testable."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(IST)


class OffsetClock:
    """Real time plus an adjustable offset, so a live demo can fast-forward (e.g. past a hold's expiry)."""

    def __init__(self) -> None:
        self.offset = timedelta()

    def now(self) -> datetime:
        return datetime.now(IST) + self.offset

    def advance(self, **kwargs: float) -> datetime:
        self.offset += timedelta(**kwargs)
        return self.now()


class FakeClock:
    """Deterministic clock for tests and scripted demos."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 11, 10, 9, 0, tzinfo=IST)

    def now(self) -> datetime:
        return self._now

    def advance(self, **kwargs: float) -> datetime:
        self._now = self._now + timedelta(**kwargs)
        return self._now

    def set(self, value: datetime) -> None:
        self._now = value


def parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=IST)


def fmt_dt(value: datetime) -> str:
    return value.astimezone(IST).isoformat(timespec="minutes")


def slot_label(value: str) -> str:
    """Human label for an ISO slot, e.g. "Wed 11 Nov, 08:00". Computed in code and handed to the
    model with every slot, so the model never has to work out a weekday itself (BUGS.md #4)."""
    return parse_dt(value).strftime("%a %d %b, %H:%M")
