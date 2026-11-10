"""Executes approved side effects (book / cancel / reschedule) correctly.

Guarantees:
  * Only runs against an approval in `approved` state whose fingerprint matches
    the current material details (see ApprovalManager.verify_material).
  * The idempotency key is derived deterministically and written to the
    action_log *before* the call, so retries, repeated confirmations and
    process restarts all reuse the same key.
  * A timeout or unexpected error is never treated as success or failure.
    get_booking_status, the source of truth, decides.
  * "confirmed" is returned only after verification.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from errand_agent.approvals import Approval, ApprovalError, ApprovalManager, MaterialDetails
from errand_agent.budget import BudgetExceeded, Limits
from errand_agent.clock import Clock, fmt_dt
from errand_agent.gateway import ToolGateway
from errand_agent.mocks.errors import ToolError
from errand_agent.store import AgentStore
from errand_agent.tracing import Tracer

ResultStatus = Literal["confirmed", "failed", "unknown", "already_done"]

# book() errors that guarantee no booking was made.
_BOOK_NO_SIDE_EFFECT = {"slot_taken", "hold_expired", "hold_not_found", "hold_already_used"}


class Hold(BaseModel):
    hold_id: str
    provider_id: str
    provider_name: str
    service: str
    slot: str
    quoted_price: int
    expires_at: str

    def details(self) -> MaterialDetails:
        return MaterialDetails(action="book", provider_id=self.provider_id, provider_name=self.provider_name,
                               service=self.service, slot=self.slot, price=self.quoted_price)


class ExecutionResult(BaseModel):
    status: ResultStatus
    action: str
    approval_id: str
    reason: str | None = None
    booking: dict | None = None
    # True when nothing material changed and the approval is still valid
    # (e.g. hold expired: re-hold the same slot at the same price and retry).
    approval_still_valid: bool = False

    @property
    def ok(self) -> bool:
        return self.status in ("confirmed", "already_done")


def book_key(approval_id: str, hold_id: str) -> str:
    return f"book:{approval_id}:{hold_id}"


class ActionExecutor:
    def __init__(self, store: AgentStore, approvals: ApprovalManager, gateway: ToolGateway,
                 clock: Clock, tracer: Tracer, limits: Limits | None = None, user_id: str = "u1") -> None:
        self.store = store
        self.approvals = approvals
        self.gateway = gateway
        self.clock = clock
        self.tracer = tracer
        self.limits = limits or Limits()
        self.user_id = user_id

    # ------------------------------------------------------------ action log
    def _log_begin(self, key: str, approval_id: str, action: str, request: dict) -> None:
        now = fmt_dt(self.clock.now())
        self.store.db.execute(
            "INSERT INTO action_log VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(idempotency_key) DO NOTHING",
            (key, approval_id, action, json.dumps(request), "pending", None, now, now))

    def _log_end(self, key: str, status: str, result: dict | None) -> None:
        self.store.db.execute(
            "UPDATE action_log SET status = ?, result = ?, updated_at = ? WHERE idempotency_key = ?",
            (status, json.dumps(result) if result is not None else None, fmt_dt(self.clock.now()), key))
        self.tracer.emit("action_log", idempotency_key=key, status=status)

    def completed_result(self, approval_id: str) -> dict | None:
        row = self.store.db.execute(
            "SELECT result FROM action_log WHERE approval_id = ? AND status = 'succeeded'", (approval_id,)).fetchone()
        return json.loads(row["result"]) if row else None

    def _start(self, approval: Approval, current: MaterialDetails, key: str, request: dict) -> None:
        """Verify material details, then atomically: approved->executing + write-ahead log row."""
        self.approvals.verify_material(approval.approval_id, current)  # invalidation commits on its own
        with self.store.tx():
            self.approvals.mark_executing(approval.approval_id)
            self._log_begin(key, approval.approval_id, current.action, request)
        self.tracer.emit("state", entity="action", action=current.action, to="EXECUTING",
                         approval_id=approval.approval_id, idempotency_key=key)

    def _precheck(self, approval_id: str, action: str) -> Approval | ExecutionResult:
        approval = self.approvals.get(approval_id)
        if approval is None:
            return ExecutionResult(status="failed", action=action, approval_id=approval_id, reason="no_such_approval")
        if approval.status == "completed":
            # Repeated confirmation: report the verified outcome, never act twice.
            return ExecutionResult(status="already_done", action=action, approval_id=approval_id,
                                   booking=self.completed_result(approval_id), reason="already_completed")
        return approval

    def _status(self, key: str, hold_id: str | None = None) -> dict | None:
        """Ask the source of truth. Returns None if it cannot be reached."""
        try:
            return self.gateway.call_with_retry(
                "get_booking_status", retries=self.limits.status_check_retries,
                backoff=self.limits.retry_backoff_seconds, essential=True,
                idempotency_key=key, **({"hold_id": hold_id} if hold_id else {}))
        except (ToolError, BudgetExceeded) as exc:
            self.tracer.emit("verification_failed", idempotency_key=key, error=str(exc))
            return None

    # ------------------------------------------------------------------ book
    def execute_book(self, approval_id: str, hold: Hold) -> ExecutionResult:
        pre = self._precheck(approval_id, "book")
        if isinstance(pre, ExecutionResult):
            return pre
        approval = pre
        if approval.status == "executing":
            # In flight already (double click, or a crash mid-call). Reconcile, don't re-send.
            return self.reconcile_book(approval_id)
        if self.clock.now() >= datetime.fromisoformat(hold.expires_at):
            return ExecutionResult(status="failed", action="book", approval_id=approval_id, reason="hold_expired",
                                   approval_still_valid=approval.status == "approved")
        key = book_key(approval_id, hold.hold_id)
        try:
            self._start(approval, hold.details(), key, {"hold": hold.model_dump()})
        except ApprovalError as exc:
            return ExecutionResult(status="failed", action="book", approval_id=approval_id, reason=exc.reason)
        return self._run_book(approval_id, hold, key)

    def _run_book(self, approval_id: str, hold: Hold, key: str, attempts_done: int = 0) -> ExecutionResult:
        attempts = attempts_done
        ambiguous: str | None = None
        while True:
            attempts += 1
            try:
                # essential: once approved, a call cap must not leave the booking half-done
                self.gateway.call("book", essential=True, hold_id=hold.hold_id, idempotency_key=key)
            except ToolError as exc:
                if exc.code in _BOOK_NO_SIDE_EFFECT:
                    return self._book_failed(approval_id, key, exc.code)
                # timeout / 503 / anything unexpected: the outcome is ambiguous.
                self.tracer.emit("ambiguous_outcome", tool="book", error=exc.code, idempotency_key=key)
                ambiguous = exc.code
            except BudgetExceeded as exc:
                # Raised before the request is sent, and any earlier attempt was
                # verified not_found, so nothing was booked.
                return self._book_failed(approval_id, key, f"budget:{exc.kind}")
            # Success or ambiguity: either way, believe only the source of truth.
            status = self._status(key, hold.hold_id)
            if status is None:
                return self._book_unknown(approval_id, key, "status_unreachable")
            if status.get("status") == "confirmed":
                result = self._book_succeeded(approval_id, key, status)
                if ambiguous:
                    result.reason = f"verified_after_{ambiguous}"
                return result
            if status.get("status") != "not_found":
                return self._book_unknown(approval_id, key, f"unexpected_status:{status.get('status')}")
            # Verified NOT booked.
            if status.get("hold_status") == "expired" or self.clock.now() >= datetime.fromisoformat(hold.expires_at):
                return self._book_failed(approval_id, key, "hold_expired")
            if attempts > self.limits.book_retries:
                return self._book_failed(approval_id, key, "retries_exhausted")
            self.tracer.emit("retry", tool="book", attempt=attempts, idempotency_key=key, reason="verified_not_booked")

    def _book_succeeded(self, approval_id: str, key: str, booking: dict) -> ExecutionResult:
        with self.store.tx():
            self._log_end(key, "succeeded", booking)
            self.approvals.complete(approval_id)
        self.tracer.emit("state", entity="action", action="book", to="CONFIRMED", booking_id=booking.get("booking_id"))
        return ExecutionResult(status="confirmed", action="book", approval_id=approval_id, booking=booking)

    def _book_failed(self, approval_id: str, key: str, reason: str) -> ExecutionResult:
        still_valid = reason in ("hold_expired", "retries_exhausted") or reason.startswith("budget:")
        with self.store.tx():
            self._log_end(key, "failed", {"reason": reason})
            if still_valid:
                self.approvals.release(approval_id, reason)
            else:
                self.approvals.invalidate(approval_id, reason)
        self.tracer.emit("state", entity="action", action="book", to="FAILED", reason=reason)
        return ExecutionResult(status="failed", action="book", approval_id=approval_id, reason=reason,
                               approval_still_valid=still_valid)

    def _book_unknown(self, approval_id: str, key: str, reason: str) -> ExecutionResult:
        # Approval stays `executing`: nobody may re-book until reconciliation decides.
        self._log_end(key, "unknown", {"reason": reason})
        self.tracer.emit("state", entity="action", action="book", to="UNKNOWN", reason=reason)
        return ExecutionResult(status="unknown", action="book", approval_id=approval_id, reason=reason)

    def reconcile_book(self, approval_id: str, resume: bool = True) -> ExecutionResult:
        """Resolve an executing/unknown booking from the source of truth (also used on restart)."""
        row = self.store.db.execute(
            "SELECT * FROM action_log WHERE approval_id = ? AND action = 'book' AND status IN ('pending','unknown') "
            "ORDER BY created_at DESC LIMIT 1", (approval_id,)).fetchone()
        if row is None:
            done = self.completed_result(approval_id)
            if done:
                return ExecutionResult(status="already_done", action="book", approval_id=approval_id, booking=done)
            return ExecutionResult(status="failed", action="book", approval_id=approval_id, reason="nothing_to_reconcile")
        key = row["idempotency_key"]
        hold = Hold.model_validate(json.loads(row["request"])["hold"])
        self.tracer.emit("reconcile", idempotency_key=key, prior_status=row["status"])
        status = self._status(key, hold.hold_id)
        if status is None:
            return self._book_unknown(approval_id, key, "status_unreachable")
        if status.get("status") == "confirmed":
            return self._book_succeeded(approval_id, key, status)
        expired = status.get("hold_status") == "expired" or self.clock.now() >= datetime.fromisoformat(hold.expires_at)
        if expired:
            return self._book_failed(approval_id, key, "hold_expired")
        if not resume:
            return self._book_failed(approval_id, key, "not_booked")
        # Verified not booked, hold still valid, approval still `executing` under
        # the same fingerprint: safe to resume with the SAME idempotency key.
        return self._run_book(approval_id, hold, key, attempts_done=1)

    # ---------------------------------------------------- cancel / reschedule
    def _verify_booking(self, booking_id: str) -> dict | None:
        try:
            data = self.gateway.call_with_retry("list_bookings", retries=self.limits.status_check_retries,
                                                backoff=self.limits.retry_backoff_seconds, essential=True,
                                                user_id=self.user_id)
        except (ToolError, BudgetExceeded):
            return None
        return next((b for b in data["bookings"] if b["booking_id"] == booking_id), None)

    def _execute_mutation(self, approval_id: str, current: MaterialDetails, tool: str, args: dict,
                          verify) -> ExecutionResult:
        pre = self._precheck(approval_id, current.action)
        if isinstance(pre, ExecutionResult):
            return pre
        key = f"{current.action}:{approval_id}"
        if pre.status != "executing":
            try:
                self._start(pre, current, key, {"tool": tool, "args": args})
            except ApprovalError as exc:
                return ExecutionResult(status="failed", action=current.action, approval_id=approval_id, reason=exc.reason)
        error: str | None = None
        try:
            self.gateway.call_with_retry(tool, retries=self.limits.tool_retries,
                                         backoff=self.limits.retry_backoff_seconds, idempotency_key=key, **args)
        except ToolError as exc:
            error = exc.code
        except BudgetExceeded as exc:
            error = f"budget:{exc.kind}"
        # Verify against the source of truth regardless of what the call said.
        booking = self._verify_booking(args["booking_id"])
        if booking is not None and verify(booking):
            with self.store.tx():
                self._log_end(key, "succeeded", booking)
                self.approvals.complete(approval_id)
            self.tracer.emit("state", entity="action", action=current.action, to="CONFIRMED", booking_id=booking["booking_id"])
            return ExecutionResult(status="confirmed", action=current.action, approval_id=approval_id, booking=booking)
        if booking is None and error is None:
            self._log_end(key, "unknown", {"reason": "verification_unavailable"})
            return ExecutionResult(status="unknown", action=current.action, approval_id=approval_id,
                                   reason="verification_unavailable")
        reason = error or "verification_mismatch"
        definitive = reason in ("slot_conflict", "not_cancellable", "not_reschedulable", "booking_not_found")
        with self.store.tx():
            self._log_end(key, "failed", {"reason": reason})
            if definitive:
                self.approvals.invalidate(approval_id, reason)
            else:
                self.approvals.release(approval_id, reason)
        self.tracer.emit("state", entity="action", action=current.action, to="FAILED", reason=reason)
        return ExecutionResult(status="failed", action=current.action, approval_id=approval_id, reason=reason,
                               approval_still_valid=not definitive)

    def execute_cancel(self, approval_id: str, booking: dict) -> ExecutionResult:
        current = MaterialDetails(action="cancel", provider_id=booking["provider_id"],
                                  provider_name=booking["provider_name"], service=booking["service"],
                                  slot=booking["slot"], price=booking["price"], booking_id=booking["booking_id"])
        return self._execute_mutation(approval_id, current, "cancel", {"booking_id": booking["booking_id"]},
                                      verify=lambda b: b["status"] == "cancelled")

    def execute_reschedule(self, approval_id: str, booking: dict, new_slot: str) -> ExecutionResult:
        current = MaterialDetails(action="reschedule", provider_id=booking["provider_id"],
                                  provider_name=booking["provider_name"], service=booking["service"],
                                  slot=new_slot, price=booking["price"], booking_id=booking["booking_id"])
        return self._execute_mutation(approval_id, current, "reschedule",
                                      {"booking_id": booking["booking_id"], "slot": new_slot},
                                      verify=lambda b: b["slot"] == new_slot and b["status"] == "confirmed")

    # --------------------------------------------------------------- restart
    def recover(self) -> list[ExecutionResult]:
        """Run on startup: settle every side effect that was in flight when we died."""
        rows = self.store.db.execute(
            "SELECT DISTINCT approval_id, action FROM action_log WHERE status IN ('pending','unknown')").fetchall()
        results = []
        for row in rows:
            self.tracer.emit("recovery", approval_id=row["approval_id"], action=row["action"])
            if row["action"] == "book":
                results.append(self.reconcile_book(row["approval_id"]))
            else:
                log = self.store.db.execute(
                    "SELECT request FROM action_log WHERE approval_id = ? AND status IN ('pending','unknown')",
                    (row["approval_id"],)).fetchone()
                req = json.loads(log["request"])
                booking = self._verify_booking(req["args"]["booking_id"])
                if booking is None:
                    continue
                # Approval is still `executing`, so this re-sends with the SAME key (idempotent).
                if row["action"] == "cancel":
                    results.append(self.execute_cancel(row["approval_id"], booking))
                else:
                    results.append(self.execute_reschedule(row["approval_id"], booking, req["args"]["slot"]))
        return results
