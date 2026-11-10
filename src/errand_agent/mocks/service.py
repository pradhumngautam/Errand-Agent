"""Mock home-services provider platform implementing the fixed tool contract.

State lives in its own SQLite file, which plays the "external system", so it
survives agent restarts. That is what makes restart-mid-booking and
timeout-after-success testable.

Tools (names/inputs/outputs/failures fixed by the assignment):
    search_providers, get_provider, hold_slot, book, get_booking_status,
    cancel, reschedule, list_bookings
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from errand_agent.clock import IST, Clock, fmt_dt, parse_dt
from errand_agent.mocks.catalog import SERVICES, Catalog
from errand_agent.mocks.errors import ToolError, ToolTimeout, TransientToolError
from errand_agent.mocks.scenario import Scenario

_SCHEMA = """
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS holds (
    hold_id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, slot TEXT NOT NULL,
    quoted_price INTEGER NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
    status TEXT NOT NULL  -- active | booked | released
);
CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, provider_id TEXT NOT NULL,
    provider_name TEXT NOT NULL, service TEXT NOT NULL, slot TEXT NOT NULL, price INTEGER NOT NULL,
    status TEXT NOT NULL,  -- confirmed | completed | cancelled | cancelled_by_provider
    hold_id TEXT, idempotency_key TEXT UNIQUE, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency (key TEXT PRIMARY KEY, op TEXT NOT NULL, result TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cancel_attempts (booking_id TEXT PRIMARY KEY, count INTEGER NOT NULL);
"""

DEFAULT_USER = "u1"


class MockProviderService:
    def __init__(self, db_path: str | Path, clock: Clock, scenario: Scenario | None = None,
                 user_id: str = DEFAULT_USER, seed_history: bool = True) -> None:
        self.clock = clock
        self.scenario = scenario or Scenario()
        self.catalog = Catalog(self.scenario.seed)
        self._rng = random.Random(self.scenario.seed)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        if seed_history and not self.db.execute("SELECT 1 FROM bookings LIMIT 1").fetchone():
            self._seed_history(user_id)

    # ------------------------------------------------------------------ helpers
    def _next_id(self, name: str, prefix: str) -> str:
        self.db.execute("INSERT INTO counters(name, value) VALUES(?, 0) ON CONFLICT(name) DO NOTHING", (name,))
        self.db.execute("UPDATE counters SET value = value + 1 WHERE name = ?", (name,))
        value = self.db.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()[0]
        return f"{prefix}-{value:04d}"

    def _maybe_fail(self, tool: str) -> None:
        queue = self.scenario.tool_failures.get(tool)
        if not queue:
            return
        code = queue.pop(0)
        if code == "timeout":
            raise ToolTimeout()
        if code.endswith("_503"):
            raise TransientToolError(code)
        raise ToolError(code, f"injected failure in {tool}")

    def _occupied(self, provider_id: str, exclude_booking: str | None = None) -> set[str]:
        now = self.clock.now().astimezone(IST).isoformat(timespec="seconds")
        rows = self.db.execute(
            "SELECT slot FROM bookings WHERE provider_id = ? AND status = 'confirmed' AND booking_id != ?",
            (provider_id, exclude_booking or ""),
        ).fetchall()
        held = self.db.execute(
            "SELECT slot FROM holds WHERE provider_id = ? AND status = 'active' AND expires_at > ?",
            (provider_id, now),
        ).fetchall()
        return {r["slot"] for r in rows} | {r["slot"] for r in held}

    def _available_slots(self, provider_id: str, start: date, end: date) -> list[str]:
        now = self.clock.now()
        occupied = self._occupied(provider_id)
        return [
            fmt_dt(s) for s in self.catalog.slots_in_range(provider_id, start, end)
            if s > now and fmt_dt(s) not in occupied
        ]

    @staticmethod
    def _booking_dict(row: sqlite3.Row) -> dict:
        return {
            "booking_id": row["booking_id"], "provider_id": row["provider_id"],
            "provider_name": row["provider_name"], "service": row["service"], "slot": row["slot"],
            "price": row["price"], "status": row["status"],
        }

    def _insert_booking(self, user_id: str, provider_id: str, slot: str, price: int, status: str,
                        hold_id: str | None, key: str | None) -> dict:
        provider = self.catalog.get(provider_id)
        assert provider is not None
        booking_id = self._next_id("booking", "BK")
        now = fmt_dt(self.clock.now())
        self.db.execute(
            "INSERT INTO bookings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (booking_id, user_id, provider_id, provider.name, provider.service, slot, price, status,
             hold_id, key, now, now),
        )
        return self._booking_dict(self.db.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone())

    def _seed_history(self, user_id: str) -> None:
        """Past and upcoming bookings so 'same electrician as last time' and 'move my AC servicing' work."""
        today = self.clock.now().date()

        def at(days: int, hour: int) -> str:
            d = today + timedelta(days=days)
            return fmt_dt(datetime(d.year, d.month, d.day, hour, tzinfo=IST))

        self._insert_booking(user_id, "PLU-KOR-03", at(-60, 10), 650, "completed", None, None)
        self._insert_booking(user_id, "ELE-KOR-01", at(-39, 11), 800, "completed", None, None)
        self._insert_booking(user_id, "ACS-KOR-01", at(3, 15), 1200, "confirmed", None, None)

    # ------------------------------------------------------------------- tools
    def search_providers(self, service: str, area: str, date_range: dict | list | tuple) -> dict:
        with self._lock:
            self._maybe_fail("search_providers")
            if service not in SERVICES:
                raise ToolError("unknown_service", f"supported: {', '.join(SERVICES)}")
            if isinstance(date_range, dict):
                start_s, end_s = date_range["start"], date_range["end"]
            else:
                start_s, end_s = date_range
            start, end = date.fromisoformat(start_s[:10]), date.fromisoformat(end_s[:10])
            if end < start or (end - start).days > 31:
                raise ToolError("invalid_date_range")
            providers = self.catalog.find(service, area)
            if not providers:
                raise ToolError("unknown_area", "no providers serve this area")
            results = []
            for p in providers:
                item = p.public()
                item["available_slots"] = self._available_slots(p.provider_id, start, end)
                results.append(item)
            return {"service": service, "area": area, "date_range": {"start": start.isoformat(), "end": end.isoformat()},
                    "providers": results}

    def get_provider(self, provider_id: str) -> dict:
        with self._lock:
            self._maybe_fail("get_provider")
            p = self.catalog.get(provider_id)
            if p is None:
                raise ToolError("provider_not_found")
            details = p.public()
            details["description"] = (f"{p.name} offers {SERVICES[p.service]['label'].lower()} in {p.area} "
                                      f"with {p.years_experience} years of experience.")
            details["reviews"] = [dict(r) for r in p.reviews]
            return details

    def hold_slot(self, provider_id: str, slot: str) -> dict:
        with self._lock:
            self._maybe_fail("hold_slot")
            p = self.catalog.get(provider_id)
            if p is None:
                raise ToolError("provider_not_found")
            slot_dt = parse_dt(slot)
            slot_s = fmt_dt(slot_dt)
            offered = {fmt_dt(s) for s in self.catalog.slots(provider_id, slot_dt.date())}
            if slot_s not in offered or slot_dt <= self.clock.now():
                raise ToolError("slot_unavailable", "provider does not offer this slot")
            if slot_s in self._occupied(provider_id):
                raise ToolError("slot_taken")
            price = p.base_price + (self.scenario.hold_price_delta if self.scenario.wants_price_delta(provider_id) else 0)
            hold_id = self._next_id("hold", "H")
            now = self.clock.now()
            expires = now + timedelta(minutes=self.scenario.hold_ttl_minutes)
            self.db.execute("INSERT INTO holds VALUES (?,?,?,?,?,?,?)",
                            (hold_id, provider_id, slot_s, price, fmt_dt(now),
                             expires.astimezone(IST).isoformat(timespec="seconds"), "active"))
            return {"hold_id": hold_id, "provider_id": provider_id, "slot": slot_s, "quoted_price": price,
                    "currency": "INR", "expires_at": expires.astimezone(IST).isoformat(timespec="seconds")}

    def _draw_book_outcome(self) -> str:
        if self.scenario.book_outcomes:
            return self.scenario.book_outcomes.pop(0)
        r, acc = self._rng.random(), 0.0
        for outcome, prob in self.scenario.book_probs.items():
            acc += prob
            if r < acc:
                return outcome
        return "success"

    def book(self, hold_id: str, idempotency_key: str, user_id: str = DEFAULT_USER) -> dict:
        with self._lock:
            # Idempotent replay: the same key always maps to the same booking.
            existing = self.db.execute("SELECT * FROM bookings WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if existing:
                return {**self._booking_dict(existing), "idempotent_replay": True}
            self._maybe_fail("book")
            hold = self.db.execute("SELECT * FROM holds WHERE hold_id = ?", (hold_id,)).fetchone()
            if hold is None:
                raise ToolError("hold_not_found")
            if hold["status"] == "booked":
                raise ToolError("hold_already_used", "this hold was already booked with another key")
            if hold["status"] == "released":
                raise ToolError("slot_taken")
            if self.clock.now() >= datetime.fromisoformat(hold["expires_at"]):
                self.db.execute("UPDATE holds SET status = 'released' WHERE hold_id = ?", (hold_id,))
                raise ToolError("hold_expired")

            outcome = self._draw_book_outcome()
            if outcome == "slot_taken":
                self.db.execute("UPDATE holds SET status = 'released' WHERE hold_id = ?", (hold_id,))
                raise ToolError("slot_taken", "slot was taken by another customer")
            if outcome == "timeout_not_booked":
                raise ToolTimeout()
            booking = self._insert_booking(user_id, hold["provider_id"], hold["slot"], hold["quoted_price"],
                                           "confirmed", hold_id, idempotency_key)
            self.db.execute("UPDATE holds SET status = 'booked' WHERE hold_id = ?", (hold_id,))
            if outcome == "timeout_booked":
                raise ToolTimeout()  # the side effect happened, but the caller never hears about it
            return booking

    def get_booking_status(self, hold_id: str | None = None, idempotency_key: str | None = None) -> dict:
        with self._lock:
            self._maybe_fail("get_booking_status")
            if not hold_id and not idempotency_key:
                raise ToolError("invalid_request", "hold_id or idempotency_key required")
            row = None
            if idempotency_key:
                row = self.db.execute("SELECT * FROM bookings WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if row is None and hold_id:
                row = self.db.execute("SELECT * FROM bookings WHERE hold_id = ?", (hold_id,)).fetchone()
            if row is not None:
                return self._booking_dict(row)
            result: dict[str, Any] = {"status": "not_found"}
            if hold_id:
                hold = self.db.execute("SELECT * FROM holds WHERE hold_id = ?", (hold_id,)).fetchone()
                if hold is not None:
                    expired = self.clock.now() >= datetime.fromisoformat(hold["expires_at"])
                    result["hold_status"] = "expired" if expired and hold["status"] == "active" else hold["status"]
            return result

    def cancel(self, booking_id: str, idempotency_key: str) -> dict:
        with self._lock:
            done = self.db.execute("SELECT result FROM idempotency WHERE key = ?", (idempotency_key,)).fetchone()
            if done:
                return {**json.loads(done["result"]), "idempotent_replay": True}
            self._maybe_fail("cancel")
            row = self.db.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
            if row is None:
                raise ToolError("booking_not_found")
            if row["status"] != "confirmed":
                raise ToolError("not_cancellable", f"booking is {row['status']}")
            if self.scenario.cancel_fail_first:
                attempts = self.db.execute("SELECT count FROM cancel_attempts WHERE booking_id = ?", (booking_id,)).fetchone()
                count = attempts["count"] if attempts else 0
                self.db.execute("INSERT INTO cancel_attempts VALUES (?, ?) ON CONFLICT(booking_id) DO UPDATE SET count = ?",
                                (booking_id, count + 1, count + 1))
                if count == 0:
                    raise TransientToolError("unavailable_503")
            self.db.execute("UPDATE bookings SET status = 'cancelled', updated_at = ? WHERE booking_id = ?",
                            (fmt_dt(self.clock.now()), booking_id))
            result = self._booking_dict(self.db.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone())
            self.db.execute("INSERT INTO idempotency VALUES (?, 'cancel', ?)", (idempotency_key, json.dumps(result)))
            return result

    def reschedule(self, booking_id: str, slot: str, idempotency_key: str) -> dict:
        with self._lock:
            done = self.db.execute("SELECT result FROM idempotency WHERE key = ?", (idempotency_key,)).fetchone()
            if done:
                return {**json.loads(done["result"]), "idempotent_replay": True}
            self._maybe_fail("reschedule")
            row = self.db.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
            if row is None:
                raise ToolError("booking_not_found")
            if row["status"] != "confirmed":
                raise ToolError("not_reschedulable", f"booking is {row['status']}")
            slot_dt = parse_dt(slot)
            slot_s = fmt_dt(slot_dt)
            offered = {fmt_dt(s) for s in self.catalog.slots(row["provider_id"], slot_dt.date())}
            if (slot_s in self.scenario.reschedule_conflict_slots or slot_s not in offered
                    or slot_s in self._occupied(row["provider_id"], exclude_booking=booking_id)
                    or slot_dt <= self.clock.now()):
                raise ToolError("slot_conflict", f"{slot_s} is not available for this provider")
            self.db.execute("UPDATE bookings SET slot = ?, updated_at = ? WHERE booking_id = ?",
                            (slot_s, fmt_dt(self.clock.now()), booking_id))
            result = self._booking_dict(self.db.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone())
            self.db.execute("INSERT INTO idempotency VALUES (?, 'reschedule', ?)", (idempotency_key, json.dumps(result)))
            return result

    def list_bookings(self, user_id: str) -> dict:
        with self._lock:
            self._maybe_fail("list_bookings")
            rows = self.db.execute("SELECT * FROM bookings WHERE user_id = ? ORDER BY slot", (user_id,)).fetchall()
            return {"user_id": user_id, "bookings": [self._booking_dict(r) for r in rows]}

    # ------------------------------------------------- environment hooks (not tools)
    def simulate_provider_cancellation(self, booking_id: str) -> None:
        """The provider cancels on their side. The agent must notice on its own."""
        with self._lock:
            self.db.execute("UPDATE bookings SET status = 'cancelled_by_provider', updated_at = ? WHERE booking_id = ?",
                            (fmt_dt(self.clock.now()), booking_id))

    def close(self) -> None:
        self.db.close()
