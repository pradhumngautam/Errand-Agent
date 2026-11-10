"""Agent-side persistence: approvals, write-ahead action log, workflows, memory."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT NOT NULL,        -- JSON MaterialDetails shown to the user
    fingerprint TEXT NOT NULL,    -- hash of the material fields
    context TEXT NOT NULL,        -- JSON non-material execution context (hold_id, expiry, ...)
    status TEXT NOT NULL,         -- pending | approved | executing | completed | invalidated
    status_reason TEXT,
    approved_via TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Write-ahead log: a row is committed BEFORE the side-effecting call is made,
-- so a crash at any point leaves enough to reconcile against the provider.
CREATE TABLE IF NOT EXISTS action_log (
    idempotency_key TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL,
    action TEXT NOT NULL,
    request TEXT NOT NULL,        -- JSON
    status TEXT NOT NULL,         -- pending | succeeded | failed | unknown
    result TEXT,                  -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS action_log_approval ON action_log(approval_id);

CREATE TABLE IF NOT EXISTS workflows (
    workflow_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class AgentStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = str(db_path)
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._depth = 0

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Re-entrant IMMEDIATE transaction (takes the write lock up front)."""
        with self._lock:
            outer = self._depth == 0
            if outer:
                self.db.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self.db
            except BaseException:
                self._depth -= 1
                if outer:
                    self.db.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if outer:
                    self.db.execute("COMMIT")

    def next_id(self, name: str, prefix: str) -> str:
        with self.tx() as db:
            db.execute("INSERT INTO counters(name, value) VALUES(?, 0) ON CONFLICT(name) DO NOTHING", (name,))
            db.execute("UPDATE counters SET value = value + 1 WHERE name = ?", (name,))
            value = db.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()[0]
        return f"{prefix}-{value:04d}"

    def close(self) -> None:
        self.db.close()
