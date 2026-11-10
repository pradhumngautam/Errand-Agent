"""Persistent user memory with provenance.

Every fact records:
  * source:    explicit (the user said it) | inferred (we guessed it from behaviour)
  * confirmed: inferred facts start unconfirmed and are never used or sent to
               the model until the user confirms them
  * status:    active | superseded (corrected) | forgotten

Corrections supersede rather than overwrite, and "forget" is a status change,
so the history can be inspected. Only `usable()` facts can reach the model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from errand_agent.clock import Clock, fmt_dt
from errand_agent.store import AgentStore
from errand_agent.tracing import Tracer

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    scope TEXT NOT NULL, source TEXT NOT NULL, confirmed INTEGER NOT NULL, status TEXT NOT NULL,
    evidence TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_user ON memories(user_id, status);
"""

Source = Literal["explicit", "inferred"]
Status = Literal["active", "superseded", "forgotten"]

# Keys the workflow understands; anything else is a free-form note.
STRUCTURED_KEYS = {"area", "earliest_time", "latest_time", "budget_max", "preferred_provider"}


class Memory(BaseModel):
    memory_id: str
    key: str
    value: str
    scope: str  # "global" or a service name
    source: Source
    confirmed: bool
    status: Status
    evidence: str | None = None
    updated_at: str

    @property
    def usable(self) -> bool:
        return self.status == "active" and (self.source == "explicit" or self.confirmed)

    def line(self) -> str:
        scope = "" if self.scope == "global" else f" [{self.scope}]"
        tag = "explicit" if self.source == "explicit" else "inferred, user-confirmed"
        return f"- {self.memory_id}: {self.key}{scope} = {self.value} ({tag})"


def normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


class MemoryStore:
    def __init__(self, store: AgentStore, clock: Clock, user_id: str = "u1") -> None:
        self.store = store
        self.clock = clock
        self.user_id = user_id
        self.tracer: Tracer | None = None
        store.db.executescript(_SCHEMA)

    def _emit(self, op: str, **data) -> None:
        if self.tracer:
            self.tracer.emit("memory", op=op, **data)

    def _row(self, r) -> Memory:
        return Memory(memory_id=r["memory_id"], key=r["key"], value=r["value"], scope=r["scope"],
                      source=r["source"], confirmed=bool(r["confirmed"]), status=r["status"],
                      evidence=r["evidence"], updated_at=r["updated_at"])

    def all(self, include_inactive: bool = True) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE user_id = ?" + ("" if include_inactive else " AND status = 'active'")
        return [self._row(r) for r in self.store.db.execute(sql + " ORDER BY memory_id", (self.user_id,)).fetchall()]

    def get(self, memory_id: str) -> Memory | None:
        r = self.store.db.execute("SELECT * FROM memories WHERE memory_id = ? AND user_id = ?",
                                  (memory_id, self.user_id)).fetchone()
        return self._row(r) if r else None

    def usable(self) -> list[Memory]:
        return [m for m in self.all(include_inactive=False) if m.usable]

    def remember(self, key: str, value: str, *, source: Source = "explicit", scope: str = "global",
                 evidence: str | None = None, confirmed: bool | None = None, created_at: str | None = None) -> Memory:
        key, scope = normalize_key(key), (scope or "global").strip().lower()
        confirmed = (source == "explicit") if confirmed is None else confirmed
        now = created_at or fmt_dt(self.clock.now())
        with self.store.tx() as db:
            # An explicit statement supersedes anything active for the same key and scope.
            # An inference never overrides an explicit fact.
            existing = db.execute(
                "SELECT * FROM memories WHERE user_id = ? AND key = ? AND scope = ? AND status = 'active'",
                (self.user_id, key, scope)).fetchall()
            if source == "inferred" and any(r["source"] == "explicit" for r in existing):
                return self._row(next(r for r in existing if r["source"] == "explicit"))
            for r in existing:
                db.execute("UPDATE memories SET status = 'superseded', updated_at = ? WHERE memory_id = ?",
                           (now, r["memory_id"]))
            memory_id = self.store.next_id("memory", "M")
            db.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (memory_id, self.user_id, key, str(value), scope, source, int(confirmed), "active",
                        evidence, now, now))
        self._emit("remember", memory_id=memory_id, key=key, scope=scope, source=source, confirmed=confirmed,
                   superseded=[r["memory_id"] for r in existing])
        return self.get(memory_id)  # type: ignore[return-value]

    def forget(self, *, memory_id: str | None = None, key: str | None = None, scope: str | None = None) -> list[str]:
        if not memory_id and not key:
            return []
        sql, params = "SELECT memory_id FROM memories WHERE user_id = ? AND status = 'active'", [self.user_id]
        if memory_id:
            sql += " AND memory_id = ?"
            params.append(memory_id)
        if key:
            sql += " AND key = ?"
            params.append(normalize_key(key))
        if scope:
            sql += " AND scope = ?"
            params.append(scope.lower())
        with self.store.tx() as db:
            ids = [r[0] for r in db.execute(sql, params).fetchall()]
            for mid in ids:
                db.execute("UPDATE memories SET status = 'forgotten', updated_at = ? WHERE memory_id = ?",
                           (fmt_dt(self.clock.now()), mid))
        self._emit("forget", memory_ids=ids)
        return ids

    def confirm(self, memory_id: str) -> Memory | None:
        m = self.get(memory_id)
        if m is None or m.status != "active":
            return None
        with self.store.tx() as db:
            db.execute("UPDATE memories SET confirmed = 1, updated_at = ? WHERE memory_id = ?",
                       (fmt_dt(self.clock.now()), memory_id))
        self._emit("confirm", memory_id=memory_id)
        return self.get(memory_id)

    def preference(self, key: str, service: str | None = None) -> Memory | None:
        """Most specific usable value: service-scoped first, then global."""
        usable = [m for m in self.usable() if m.key == key]
        for scope in ([service] if service else []) + ["global"]:
            hit = [m for m in usable if m.scope == scope]
            if hit:
                return hit[-1]
        return None
