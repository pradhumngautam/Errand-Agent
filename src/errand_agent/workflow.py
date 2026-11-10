"""Explicit workflow state, persisted per user.

The model never sets the phase. Code moves the phase in response to tool
outcomes, and every move is validated against TRANSITIONS and traced. The key
structural invariant: EXECUTING is reachable only from AWAITING_APPROVAL.
"""

from __future__ import annotations

import json
import re
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from errand_agent.clock import Clock, fmt_dt
from errand_agent.executor import Hold
from errand_agent.memory import MemoryStore
from errand_agent.store import AgentStore
from errand_agent.tracing import Tracer


class Phase(str, Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"            # required details missing
    PRESENTING = "PRESENTING"            # options shown, user choosing
    HOLDING = "HOLDING"                  # a slot is held, approval not yet requested
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTING = "EXECUTING"              # side effect in flight / being verified
    DONE = "DONE"
    FAILED = "FAILED"


P = Phase
_OPEN = {P.COLLECTING, P.PRESENTING, P.HOLDING, P.AWAITING_APPROVAL}
TRANSITIONS: dict[Phase, set[Phase]] = {
    P.IDLE: _OPEN | {P.IDLE},
    P.COLLECTING: _OPEN | {P.IDLE},
    P.PRESENTING: _OPEN | {P.IDLE},
    P.HOLDING: _OPEN | {P.IDLE},
    P.AWAITING_APPROVAL: _OPEN | {P.EXECUTING, P.IDLE},
    P.EXECUTING: {P.DONE, P.FAILED, P.PRESENTING, P.HOLDING, P.AWAITING_APPROVAL, P.EXECUTING},
    P.DONE: _OPEN | {P.IDLE},
    P.FAILED: _OPEN | {P.IDLE},
}


class IllegalTransition(Exception):
    pass


_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class ErrandRequest(BaseModel):
    intent: Literal["book", "reschedule", "cancel"] | None = None
    service: str | None = None
    area: str | None = None
    date_start: str | None = None      # YYYY-MM-DD
    date_end: str | None = None
    earliest_time: str | None = None   # HH:MM
    latest_time: str | None = None
    budget_max: int | None = None
    no_budget_limit: bool = False
    provider_id: str | None = None     # a specific provider the user asked for
    booking_id: str | None = None      # for reschedule / cancel
    new_slot: str | None = None        # for reschedule
    sources: dict[str, str] = Field(default_factory=dict)  # field -> "user" | "memory:<id>"

    def validate_fields(self) -> list[str]:
        problems = []
        for f in ("earliest_time", "latest_time"):
            v = getattr(self, f)
            if v and not _TIME.match(v):
                problems.append(f"{f} must be HH:MM")
        for f in ("date_start", "date_end"):
            v = getattr(self, f)
            if v:
                try:
                    date.fromisoformat(v)
                except ValueError:
                    problems.append(f"{f} must be YYYY-MM-DD")
        if self.date_start and self.date_end and not problems and self.date_end < self.date_start:
            problems.append("date_end is before date_start")
        if self.budget_max is not None and self.budget_max <= 0:
            problems.append("budget_max must be positive")
        return problems

    def effective(self, memory: MemoryStore) -> "ErrandRequest":
        """Fill unset fields from usable (explicit or confirmed) memory. Never overrides the user."""
        eff = self.model_copy(deep=True)
        mapping = {"area": "area", "earliest_time": "earliest_time", "latest_time": "latest_time",
                   "budget_max": "budget_max"}
        for field, key in mapping.items():
            if getattr(eff, field) is None and not (field == "budget_max" and eff.no_budget_limit):
                mem = memory.preference(key, eff.service)
                if mem is not None:
                    value: object = mem.value
                    if field == "budget_max":
                        try:
                            value = int(float(mem.value))
                        except ValueError:
                            continue
                    setattr(eff, field, value)
                    eff.sources[field] = f"memory:{mem.memory_id}"
        if eff.provider_id is None and eff.intent == "book" and eff.service:
            mem = memory.preference("preferred_provider", eff.service)
            if mem is not None:
                eff.sources["preferred_provider"] = f"memory:{mem.memory_id}"
        return eff

    def missing(self) -> list[str]:
        """Required details still unknown, for the current intent."""
        if self.intent in (None, "book"):
            need = []
            if not self.service:
                need.append("service")
            if not self.area:
                need.append("area")
            if not self.date_start or not self.date_end:
                need.append("date_window")
            if self.budget_max is None and not self.no_budget_limit:
                need.append("budget")
            return need
        if self.intent == "reschedule":
            return [n for n, v in (("which_booking", self.booking_id or self.service), ("new_date_or_slot",
                    self.new_slot or self.date_start)) if not v]
        if self.intent == "cancel":
            return [] if (self.booking_id or self.service) else ["which_booking"]
        return []


class WorkflowState(BaseModel):
    workflow_id: str
    user_id: str
    origin: Literal["user", "proactive"] = "user"
    phase: Phase = Phase.IDLE
    request: ErrandRequest = Field(default_factory=ErrandRequest)
    candidates: list[dict] = Field(default_factory=list)
    hold: Hold | None = None
    target_booking: dict | None = None
    pending_approval_id: str | None = None
    last_completed_approval_id: str | None = None
    last_outcome: dict | None = None

    def transition(self, to: Phase, tracer: Tracer, reason: str = "") -> None:
        if to not in TRANSITIONS[self.phase]:
            tracer.emit("illegal_transition", frm=self.phase.value, to=to.value, reason=reason)
            raise IllegalTransition(f"{self.phase.value} -> {to.value}")
        if to != self.phase:
            tracer.emit("state", entity="workflow", workflow_id=self.workflow_id, frm=self.phase.value,
                        to=to.value, reason=reason)
        self.phase = to

    def summary_for_model(self, effective: ErrandRequest) -> str:
        req = {k: v for k, v in effective.model_dump(exclude={"sources"}).items() if v not in (None, False)}
        lines = [f"phase: {self.phase.value}", f"request: {json.dumps(req)}"]
        if effective.sources:
            lines.append(f"field sources: {json.dumps(effective.sources)}")
        lines.append(f"missing details: {effective.missing() or 'none'}")
        if self.candidates:
            lines.append("shortlist: " + json.dumps(self.candidates[:5]))
        if self.hold:
            lines.append("current hold: " + self.hold.model_dump_json())
        if self.target_booking:
            lines.append("target booking: " + json.dumps(self.target_booking))
        if self.pending_approval_id:
            lines.append(f"pending approval: {self.pending_approval_id} (waiting for the user)")
        if self.last_outcome:
            lines.append("last outcome: " + json.dumps(self.last_outcome))
        return "\n".join(lines)


class WorkflowRepo:
    def __init__(self, store: AgentStore, clock: Clock) -> None:
        self.store = store
        self.clock = clock

    def active(self, user_id: str) -> WorkflowState:
        row = self.store.db.execute(
            "SELECT data FROM workflows WHERE user_id = ? ORDER BY rowid DESC LIMIT 1", (user_id,)).fetchone()
        if row:
            return WorkflowState.model_validate_json(row["data"])
        return self.new(user_id)

    def new(self, user_id: str, origin: Literal["user", "proactive"] = "user") -> WorkflowState:
        wf = WorkflowState(workflow_id=self.store.next_id("workflow", "WF"), user_id=user_id, origin=origin)
        self.save(wf)
        return wf

    def save(self, wf: WorkflowState) -> None:
        with self.store.tx() as db:
            db.execute("INSERT INTO workflows VALUES (?,?,?,?,?) ON CONFLICT(workflow_id) DO UPDATE SET "
                       "state = excluded.state, data = excluded.data, updated_at = excluded.updated_at",
                       (wf.workflow_id, wf.user_id, wf.phase.value, wf.model_dump_json(), fmt_dt(self.clock.now())))
