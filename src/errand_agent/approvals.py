"""Approval enforcement in application code.

The model can only *request* an approval (which renders a card for the user).
Approval is granted only by:
  * the UI confirm button, which carries the approval_id, or
  * a chat message that `is_explicit_confirmation` accepts while exactly one
    approval is pending.
Execution re-checks the fingerprint of the material details (action,
provider, slot, price, booking) against the approved one, so any change to
provider, slot or price invalidates the approval.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from errand_agent.clock import Clock, fmt_dt, parse_dt
from errand_agent.store import AgentStore
from errand_agent.tracing import Tracer

Action = Literal["book", "cancel", "reschedule"]
ApprovalStatus = Literal["pending", "approved", "executing", "completed", "invalidated"]


class MaterialDetails(BaseModel):
    """Everything the user is agreeing to. All fields are part of the fingerprint."""

    action: Action
    provider_id: str
    provider_name: str
    service: str
    slot: str
    price: int
    booking_id: str | None = None

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def summary(self) -> str:
        verb = {"book": "Book", "cancel": "Cancel", "reschedule": "Reschedule"}[self.action]
        target = f" (booking {self.booking_id})" if self.booking_id else ""
        when = "to " if self.action == "reschedule" else "on "
        try:
            slot = parse_dt(self.slot).strftime("%a %d %b, %H:%M")
        except ValueError:
            slot = self.slot
        return f"{verb} {self.provider_name}{target} {when}{slot} for ₹{self.price}"


class Approval(BaseModel):
    approval_id: str
    workflow_id: str
    details: MaterialDetails
    fingerprint: str
    context: dict = Field(default_factory=dict)
    status: ApprovalStatus
    status_reason: str | None = None
    approved_via: str | None = None


class ApprovalError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_AFFIRM = r"(?:yes|yeah|yep|yup|y|sure|ok|okay|confirm(?:ed)?|approve(?:d)?|go ahead|proceed|do it|book it|" \
          r"cancel it|reschedule it|move it|please do|sounds good|that'?s fine)"
_CONFIRM_RE = re.compile(rf"^\s*{_AFFIRM}(?:[\s,!.]+(?:please|{_AFFIRM}|thanks|thank you))*[\s!.]*$", re.IGNORECASE)


def is_explicit_confirmation(text: str) -> bool:
    """True only for a bare affirmative. "yes but at 11" or "ok what about X" are NOT approvals."""
    return bool(_CONFIRM_RE.match(text or ""))


class ApprovalManager:
    def __init__(self, store: AgentStore, clock: Clock, tracer: Tracer) -> None:
        self.store = store
        self.clock = clock
        self.tracer = tracer

    # ----------------------------------------------------------------- reads
    def _row_to_approval(self, row) -> Approval:
        return Approval(
            approval_id=row["approval_id"], workflow_id=row["workflow_id"],
            details=MaterialDetails.model_validate_json(row["details"]), fingerprint=row["fingerprint"],
            context=json.loads(row["context"]), status=row["status"], status_reason=row["status_reason"],
            approved_via=row["approved_via"],
        )

    def get(self, approval_id: str) -> Approval | None:
        row = self.store.db.execute("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone()
        return self._row_to_approval(row) if row else None

    def pending(self, workflow_id: str) -> list[Approval]:
        rows = self.store.db.execute(
            "SELECT * FROM approvals WHERE workflow_id = ? AND status = 'pending' ORDER BY created_at",
            (workflow_id,)).fetchall()
        return [self._row_to_approval(r) for r in rows]

    # ---------------------------------------------------------------- writes
    def _set(self, approval_id: str, status: ApprovalStatus, reason: str | None = None,
             expect: tuple[str, ...] | None = None, approved_via: str | None = None) -> bool:
        sql = "UPDATE approvals SET status = ?, status_reason = ?, updated_at = ?"
        params: list = [status, reason, fmt_dt(self.clock.now())]
        if approved_via:
            sql += ", approved_via = ?"
            params.append(approved_via)
        sql += " WHERE approval_id = ?"
        params.append(approval_id)
        if expect:
            sql += f" AND status IN ({','.join('?' * len(expect))})"
            params.extend(expect)
        with self.store.tx() as db:
            changed = db.execute(sql, params).rowcount == 1
        if changed:
            self.tracer.emit("approval", approval_id=approval_id, status=status, reason=reason, via=approved_via)
        return changed

    def request(self, workflow_id: str, details: MaterialDetails, context: dict | None = None) -> Approval:
        """Create a pending approval and supersede any older open one in this workflow."""
        with self.store.tx() as db:
            open_ids = [r[0] for r in db.execute(
                "SELECT approval_id FROM approvals WHERE workflow_id = ? AND status IN ('pending','approved')",
                (workflow_id,)).fetchall()]
            for old in open_ids:
                self._set(old, "invalidated", "superseded")
            approval_id = self.store.next_id("approval", "AP")
            now = fmt_dt(self.clock.now())
            db.execute(
                "INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (approval_id, workflow_id, details.action, details.model_dump_json(), details.fingerprint(),
                 json.dumps(context or {}), "pending", None, None, now, now),
            )
        self.tracer.emit("approval", approval_id=approval_id, status="pending", details=details.model_dump(),
                         fingerprint=details.fingerprint())
        return self.get(approval_id)  # type: ignore[return-value]

    def approve(self, approval_id: str, via: Literal["button", "chat"]) -> Approval:
        if not self._set(approval_id, "approved", expect=("pending",), approved_via=via):
            current = self.get(approval_id)
            raise ApprovalError(f"not_pending:{current.status if current else 'missing'}")
        return self.get(approval_id)  # type: ignore[return-value]

    def invalidate(self, approval_id: str, reason: str) -> None:
        self._set(approval_id, "invalidated", reason, expect=("pending", "approved", "executing"))

    def update_context(self, approval_id: str, context: dict) -> None:
        with self.store.tx() as db:
            db.execute("UPDATE approvals SET context = ? WHERE approval_id = ?", (json.dumps(context), approval_id))

    def verify_material(self, approval_id: str, current: MaterialDetails) -> Approval:
        """Check that what we are about to do is exactly what was approved.
        On mismatch, invalidate (committed on its own) and raise."""
        approval = self.get(approval_id)
        if approval is None:
            raise ApprovalError("missing")
        if approval.status != "approved":
            raise ApprovalError(f"not_approved:{approval.status}")
        if current.fingerprint() != approval.fingerprint:
            before = approval.details.model_dump()
            changed = [k for k, v in current.model_dump().items() if before.get(k) != v]
            self.invalidate(approval_id, f"material_change:{','.join(changed)}")
            raise ApprovalError(f"material_change:{','.join(changed)}")
        return approval

    def mark_executing(self, approval_id: str) -> None:
        """Atomic approved -> executing flip; loses any race against a second confirmation."""
        if not self._set(approval_id, "executing", expect=("approved",)):
            raise ApprovalError("race:not_approved")

    def complete(self, approval_id: str) -> None:
        self._set(approval_id, "completed", expect=("executing",))

    def release(self, approval_id: str, reason: str) -> None:
        """Execution failed with no side effect and nothing material changed
        (e.g. the hold expired). The approval stays valid for the same details."""
        self._set(approval_id, "approved", reason, expect=("executing",))
