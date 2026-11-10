"""The orchestrator.

One user message = one run with its own trace and hard limits. A run is one of:
  * approval fast-path: the user confirmed a pending approval card, so code
    executes it. The model is not involved and the reply is built from the
    verified result.
  * model loop: the model calls tools (never side-effecting ones) until it
    replies or requests an approval.
Proactive runs (poll_events) detect provider cancellations and settle
in-flight actions, without the user saying anything.
"""

from __future__ import annotations

import math
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from errand_agent.approvals import ApprovalError, ApprovalManager, is_explicit_confirmation
from errand_agent.budget import Budget, BudgetExceeded, Limits
from errand_agent.clock import Clock, fmt_dt, parse_dt, slot_label
from errand_agent.context import ContextBuilder, ContextOverflow
from errand_agent.executor import ActionExecutor, ExecutionResult, Hold
from errand_agent.gateway import ToolGateway
from errand_agent.llm.base import LLMClient, LLMError, Message, estimate_cost
from errand_agent.memory import MemoryStore
from errand_agent.mocks.errors import ToolError
from errand_agent.prompts import system_prompt
from errand_agent.reviews import HeuristicReviewAnalyzer
from errand_agent.store import AgentStore
from errand_agent.tools import TOOL_SPECS, AgentToolbox
from errand_agent.tracing import Tracer
from errand_agent.workflow import ErrandRequest, Phase, WorkflowRepo, WorkflowState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, role TEXT NOT NULL,
    text TEXT NOT NULL, run_id TEXT, ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS booking_watch (booking_id TEXT PRIMARY KEY, status TEXT NOT NULL);
"""

# First-person claims of having acted. Describing existing bookings is fine.
_CLAIM_RE = re.compile(
    r"\b(i'?ve|i have|i)\s+(just\s+|now\s+|successfully\s+)?(booked|cancell?ed|rescheduled|confirmed your)\b"
    r"|\byou'?re (all )?(booked|set)\b|\bsuccessfully (booked|cancell?ed|rescheduled)\b", re.I)


human_slot = slot_label


class AgentConfig(BaseModel):
    user_id: str = "u1"
    context_budget_tokens: int = 10000
    history_turns_considered: int = 200
    limits: Limits = Limits()
    trace_dir: str | None = None


class AgentReply(BaseModel):
    run_id: str
    text: str
    kind: str = "message"            # message | notification
    approval_card: dict | None = None
    outcome: dict | None = None       # ExecutionResult when an action ran
    phase: str = ""
    trace_path: str | None = None


class Run:
    def __init__(self, agent: "Agent", kind: str, quiet: bool = False,
                 on_event: Callable[[dict], None] | None = None) -> None:
        # background polls get their own id series so they don't eat user-visible run numbers
        self.run_id = agent.store.next_id("watch", "W") if quiet else agent.store.next_id("run", "R")
        path = Path(agent.config.trace_dir) / f"{self.run_id}.jsonl" if agent.config.trace_dir else None
        self.tracer = Tracer(self.run_id, path, listener=agent.trace_listener, quiet=quiet)
        self.on_event = on_event
        if on_event:
            self.tracer.listeners.append(on_event)
        self.budget = Budget(agent.config.limits, monotonic=agent.monotonic)
        self.gateway = ToolGateway(agent.service, self.tracer, self.budget, sleep=agent.sleep)
        self.approvals = ApprovalManager(agent.store, agent.clock, self.tracer)
        self.executor = ActionExecutor(agent.store, self.approvals, self.gateway, agent.clock, self.tracer,
                                       agent.config.limits, user_id=agent.config.user_id)
        self.toolbox: AgentToolbox | None = None
        self.tokens = {"input": 0, "output": 0}
        self.cost_usd = 0.0
        agent.memory.tracer = self.tracer
        self.tracer.emit("run_start", kind=kind)


class Agent:
    def __init__(self, *, store: AgentStore, service: Any, llm: LLMClient, clock: Clock,
                 config: AgentConfig | None = None, analyzer: Any = None,
                 sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
                 trace_listener: Callable[[dict], None] | None = None) -> None:
        self.store = store
        self.service = service
        self.llm = llm
        self.clock = clock
        self.config = config or AgentConfig()
        self.analyzer = analyzer or HeuristicReviewAnalyzer()
        self.sleep = sleep
        self.monotonic = monotonic
        self.trace_listener = trace_listener
        self.user_id = self.config.user_id
        self.memory = MemoryStore(store, clock, self.user_id)
        self.workflows = WorkflowRepo(store, clock)
        self.context = ContextBuilder(self.memory, self.config.context_budget_tokens)
        store.db.executescript(_SCHEMA)
        self.last_run: Run | None = None

    # ------------------------------------------------------------ plumbing
    def _run(self, kind: str, quiet: bool = False, on_event: Callable[[dict], None] | None = None) -> Run:
        run = Run(self, kind, quiet=quiet, on_event=on_event)
        self.last_run = run
        return run

    def _log_turn(self, role: str, text: str, run: Run) -> None:
        with self.store.tx() as db:
            db.execute("INSERT INTO conversation(user_id, role, text, run_id, ts) VALUES (?,?,?,?,?)",
                       (self.user_id, role, text, run.run_id, fmt_dt(self.clock.now())))

    def history(self, before_run: str | None = None) -> list[Message]:
        rows = self.store.db.execute(
            "SELECT role, text, run_id FROM conversation WHERE user_id = ? ORDER BY seq DESC LIMIT ?",
            (self.user_id, self.config.history_turns_considered)).fetchall()
        msgs = [Message(role="user" if r["role"] == "user" else "assistant", content=r["text"])
                for r in reversed(rows) if r["run_id"] != before_run]
        return msgs

    def _finish(self, run: Run, state: WorkflowState, text: str, card: dict | None = None,
                outcome: ExecutionResult | None = None, kind: str = "message") -> AgentReply:
        self.workflows.save(state)
        self._log_turn("assistant", text, run)
        run.tracer.emit("run_end", phase=state.phase.value, reply=text, approval_card=card,
                        outcome=outcome.model_dump() if outcome else None, budget=run.budget.snapshot(),
                        tokens=run.tokens, est_cost_usd=round(run.cost_usd, 6))
        return AgentReply(run_id=run.run_id, text=text, kind=kind, approval_card=card,
                          outcome=outcome.model_dump() if outcome else None, phase=state.phase.value,
                          trace_path=str(run.tracer.path) if run.tracer.path else None)

    def _toolbox(self, run: Run, state: WorkflowState, user_text: str) -> AgentToolbox:
        run.toolbox = AgentToolbox(state=state, gateway=run.gateway, approvals=run.approvals, memory=self.memory,
                                   analyzer=self.analyzer, clock=self.clock, tracer=run.tracer,
                                   user_id=self.user_id, user_text=user_text,
                                   new_workflow=lambda: self.workflows.new(self.user_id))
        return run.toolbox

    # ------------------------------------------------------------- startup
    def startup(self) -> list[AgentReply]:
        """Settle anything in flight from a previous process, then start watching bookings."""
        replies = self._reconcile_in_flight(kind="startup_recovery")
        if not self.store.db.execute("SELECT 1 FROM booking_watch LIMIT 1").fetchone():
            run = self._run("watch_init", quiet=True)
            self._snapshot(run.gateway.call("list_bookings", user_id=self.user_id)["bookings"])
        return replies

    def _snapshot(self, bookings: list[dict]) -> None:
        with self.store.tx() as db:
            for b in bookings:
                db.execute("INSERT INTO booking_watch VALUES (?, ?) ON CONFLICT(booking_id) DO UPDATE SET status = ?",
                           (b["booking_id"], b["status"], b["status"]))

    # --------------------------------------------------------- user turns
    def handle_message(self, text: str, on_event: Callable[[dict], None] | None = None) -> AgentReply:
        """`on_event` receives every trace event of this run live, plus streamed model
        output as {"type": "llm_stream", "kind": "thought"|"text"|"reset", "chunk": ...}."""
        run = self._run("user_message", on_event=on_event)
        run.tracer.emit("user_message", text=text)
        self._log_turn("user", text, run)
        state = self.workflows.active(self.user_id)
        try:
            if state.pending_approval_id:
                if is_explicit_confirmation(text):
                    return self._confirm(run, state, state.pending_approval_id, via="chat")
                run.approvals.invalidate(state.pending_approval_id, "user_replied_without_confirming")
                state.pending_approval_id = None
                state.transition(Phase.HOLDING if state.hold else Phase.PRESENTING, run.tracer,
                                 reason="approval withdrawn: user said something else")
            elif state.last_completed_approval_id and state.phase == Phase.DONE and is_explicit_confirmation(text):
                return self._already_done(run, state)
            reply, card = self._model_loop(run, state, text)
            state = run.toolbox.state if run.toolbox else state
            return self._finish(run, state, reply, card)
        except BudgetExceeded as exc:
            state = run.toolbox.state if run.toolbox else state
            run.tracer.emit("limit_reached", kind=exc.kind, limit=exc.limit)
            return self._finish(run, state, self._limit_message(state, exc))
        except ContextOverflow as exc:
            state = run.toolbox.state if run.toolbox else state
            run.tracer.emit("context_overflow", error=str(exc))
            return self._finish(run, state, "That request needs more context than I'm allowed to use at once. "
                                "Could you narrow it down (one service, a shorter date range)? Nothing was booked.")
        except LLMError as exc:
            state = run.toolbox.state if run.toolbox else state
            run.tracer.emit("error", source="llm", error=str(exc))
            return self._finish(run, state, "I can't reach the language model right now, so I've paused here. "
                                "Nothing was booked or changed. " + self._where_we_are(state))
        except Exception as exc:  # noqa: BLE001 - every exit path must produce a user-facing outcome
            state = run.toolbox.state if run.toolbox else state
            run.tracer.emit("error", source="agent", error=f"{type(exc).__name__}: {exc}")
            return self._finish(run, state, "Something went wrong on my side. Nothing was booked or changed "
                                "without your confirmation. " + self._where_we_are(state))

    def confirm(self, approval_id: str, on_event: Callable[[dict], None] | None = None) -> AgentReply:
        """The UI's confirm button (carries the exact approval id)."""
        run = self._run("approval_button", on_event=on_event)
        run.tracer.emit("user_action", action="confirm_button", approval_id=approval_id)
        self._log_turn("user", f"[confirmed {approval_id}]", run)
        state = self.workflows.active(self.user_id)
        try:
            return self._confirm(run, state, approval_id, via="button")
        except Exception as exc:  # noqa: BLE001
            run.tracer.emit("error", source="agent", error=f"{type(exc).__name__}: {exc}")
            return self._finish(run, state, "Something went wrong while confirming. " + self._where_we_are(state))

    def _model_loop(self, run: Run, state: WorkflowState, text: str) -> tuple[str, dict | None]:
        toolbox = self._toolbox(run, state, text)
        history = self.history(before_run=run.run_id)
        run_messages = [Message(role="user", content=text)]
        while True:
            run.budget.charge_model_turn()
            st = toolbox.state
            eff = st.request.effective(self.memory)
            system, messages, manifest = self.context.build(
                system_prompt(self.clock.now()), st.summary_for_model(eff), eff.service, history, run_messages,
                TOOL_SPECS)
            run.tracer.emit("context", turn=run.budget.model_turns, **manifest.model_dump())
            run.tracer.emit("model_request", turn=run.budget.model_turns, model=getattr(self.llm, "model", "?"),
                            messages=len(messages))
            stream = None
            if run.on_event:
                turn = run.budget.model_turns

                def stream(kind: str, chunk: str, _turn: int = turn) -> None:
                    run.on_event({"type": "llm_stream", "run_id": run.run_id, "turn": _turn,  # type: ignore[misc]
                                  "kind": kind, "chunk": chunk})
            with run.tracer.timed() as t:
                resp = self.llm.complete(system, messages, TOOL_SPECS, on_stream=stream)
            # calibrate only against providers whose token counts reflect our prompt alone
            if resp.model != "scripted" and getattr(self.llm, "calibrate_tokens", True):
                self.context.observe(manifest.raw_estimate, resp.usage.input_tokens)
            cost = resp.cost_usd if resp.cost_usd is not None else estimate_cost(resp.model, resp.usage)
            run.tokens["input"] += resp.usage.input_tokens
            run.tokens["output"] += resp.usage.output_tokens
            run.cost_usd += cost or 0.0
            run.tracer.emit("model_call", turn=run.budget.model_turns, model=resp.model,
                            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                            est_cost_usd=cost, latency_ms=t["latency_ms"], text=resp.text,
                            thoughts=resp.thoughts,
                            tool_calls=[{"name": c.name, "args": c.args} for c in resp.tool_calls])
            if not resp.tool_calls:
                return self._guard_claims(run, resp.text or "Sorry, I didn't catch that. Could you rephrase?"), None
            run_messages.append(Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls,
                                        provider_raw=resp.provider_raw))
            for call in resp.tool_calls:
                result = toolbox.dispatch(call.name, call.args)
                if call.name in ("update_request", "request_approval", "remember_preference", "forget_preference",
                                 "confirm_memory", "list_memories") or "error" in result:
                    run.tracer.emit("agent_tool", tool=call.name, args=call.args, result=result)
                run_messages.append(Message(role="tool", tool_call_id=call.id, name=call.name, result=result))
                if toolbox.approval_card:
                    break
            if toolbox.approval_card:
                card = toolbox.approval_card
                lead = self._guard_claims(run, resp.text) + "\n\n" if resp.text else ""
                return lead + self._card_text(card, toolbox.state), card

    # ------------------------------------------------------- confirmation
    def _card_text(self, card: dict, state: WorkflowState) -> str:
        expiry = ""
        if card["action"] == "book" and state.hold:
            expiry = f" (slot held until {parse_dt(state.hold.expires_at).strftime('%H:%M')})"
        return f"Please confirm: {card['summary']}{expiry}. Reply \"yes\" to confirm, or tell me what to change."

    def _already_done(self, run: Run, state: WorkflowState) -> AgentReply:
        booking = run.executor.completed_result(state.last_completed_approval_id or "")
        run.tracer.emit("guard", rule="repeated_confirmation", approval_id=state.last_completed_approval_id)
        text = "That's already done, and I haven't repeated it."
        if booking:
            text += f" Booking {booking['booking_id']}: {booking['provider_name']}, {human_slot(booking['slot'])}, " \
                    f"status {booking['status']}."
        return self._finish(run, state, text)

    def _confirm(self, run: Run, state: WorkflowState, approval_id: str, via: str) -> AgentReply:
        approval = run.approvals.get(approval_id)
        if approval is None or approval.workflow_id != state.workflow_id:
            return self._finish(run, state, "That confirmation doesn't match anything I'm waiting on, so I "
                                "haven't done anything.")
        if approval.status == "completed":
            return self._already_done(run, state)
        try:
            run.approvals.approve(approval_id, via=via)  # type: ignore[arg-type]
        except ApprovalError:
            return self._finish(run, state, f"That approval is no longer valid ({approval.status_reason or approval.status}"
                                "), so nothing was done. " + self._where_we_are(state))
        state.pending_approval_id = None
        state.transition(Phase.EXECUTING, run.tracer, reason=f"approved {approval_id} via {via}")
        self.workflows.save(state)  # persist EXECUTING before the side effect, so recovery can find it
        action = approval.details.action
        if action == "book":
            hold = state.hold or Hold.model_validate(approval.context["hold"])
            result = run.executor.execute_book(approval_id, hold)
            return self._after_book(run, state, result, hold)
        booking = state.target_booking or approval.context["booking"]
        if action == "cancel":
            result = run.executor.execute_cancel(approval_id, booking)
        else:
            result = run.executor.execute_reschedule(approval_id, booking, approval.context["new_slot"])
        return self._after_mutation(run, state, result, booking)

    def _after_book(self, run: Run, state: WorkflowState, result: ExecutionResult, hold: Hold,
                    rehold_attempted: bool = False) -> AgentReply:
        state.last_outcome = result.model_dump(include={"status", "reason", "action"})
        if result.ok:
            b = result.booking or {}
            state.transition(Phase.DONE, run.tracer, reason="booking verified")
            state.hold = None
            state.last_completed_approval_id = result.approval_id
            if b.get("booking_id"):
                self._snapshot([{"booking_id": b["booking_id"], "status": "confirmed"}])
            self.memory.remember("preferred_provider", b.get("provider_id", hold.provider_id), source="inferred",
                                 scope=hold.service, evidence=f"booked {b.get('booking_id')}")
            text = (f"Booked and verified with the provider: {b.get('provider_name', hold.provider_name)}, "
                    f"{human_slot(b.get('slot', hold.slot))}, ₹{b.get('price', hold.quoted_price)}. "
                    f"Booking ID {b.get('booking_id')}.")
            if result.status == "already_done":
                text = "That was already booked, and I haven't booked it again. " + text
            elif (result.reason or "").startswith("verified_after_"):
                text = ("The provider's system timed out on the booking request, so I checked the booking "
                        "status before telling you anything. It did go through, and I didn't send it twice. ") + text
            return self._finish(run, state, text, outcome=result)
        if result.status == "unknown":
            text = ("I sent the booking but couldn't verify the result with the provider yet. I won't send it "
                    "again. I'll keep checking and tell you the moment it's confirmed either way.")
            return self._finish(run, state, text, outcome=result)
        reason = result.reason or "unknown"
        if reason == "hold_expired" and result.approval_still_valid and not rehold_attempted:
            return self._rehold_after_expiry(run, state, result, hold)
        if reason == "slot_taken":
            state.hold = None
            state.transition(Phase.PRESENTING, run.tracer, reason="slot taken")
            return self._finish(run, state, "That slot was taken by someone else just before booking. Nothing was "
                                "booked. " + self._alternatives(state, exclude_slot=hold.slot), outcome=result)
        if reason.startswith("material_change"):
            fields = reason.split(":", 1)[1]
            return self._reapprove(run, state, result, f"Something changed since you approved ({fields}), so "
                                   "that approval is void and nothing was booked.")
        state.transition(Phase.HOLDING if state.hold else Phase.PRESENTING, run.tracer, reason=reason)
        return self._finish(run, state, f"I couldn't complete the booking ({reason.replace('_', ' ')}). I checked "
                            "with the provider and nothing was booked. Want me to try again or pick another slot?",
                            outcome=result)

    def _rehold_after_expiry(self, run: Run, state: WorkflowState, result: ExecutionResult, old: Hold) -> AgentReply:
        try:
            h = run.gateway.call("hold_slot", provider_id=old.provider_id, slot=old.slot)
        except ToolError as exc:
            state.hold = None
            state.transition(Phase.PRESENTING, run.tracer, reason=f"re-hold failed: {exc.code}")
            return self._finish(run, state, "Your hold expired and the slot is no longer available. Nothing was "
                                "booked. " + self._alternatives(state, exclude_slot=old.slot), outcome=result)
        new = old.model_copy(update={"hold_id": h["hold_id"], "quoted_price": h["quoted_price"],
                                     "expires_at": h["expires_at"]})
        state.hold = new
        if new.quoted_price == old.quoted_price:
            run.tracer.emit("decision", what="rehold_same_price_reuse_approval", approval_id=result.approval_id)
            again = run.executor.execute_book(result.approval_id, new)
            return self._after_book(run, state, again, new, rehold_attempted=True)
        state.transition(Phase.HOLDING, run.tracer, reason="re-hold price changed")
        return self._reapprove(run, state, result, f"Your hold expired. When I re-held the slot, the provider "
                               f"quoted ₹{new.quoted_price} instead of ₹{old.quoted_price}. Nothing was booked.")

    def _reapprove(self, run: Run, state: WorkflowState, result: ExecutionResult, why: str) -> AgentReply:
        if state.phase == Phase.EXECUTING:
            state.transition(Phase.HOLDING, run.tracer, reason="needs fresh approval")
        toolbox = self._toolbox(run, state, "")
        out = toolbox.t_request_approval("book")
        if "error" in out:
            return self._finish(run, state, why + " " + self._where_we_are(state), outcome=result)
        card = toolbox.approval_card
        return self._finish(run, state, why + "\n\n" + self._card_text(card, state), card=card, outcome=result)

    def _after_mutation(self, run: Run, state: WorkflowState, result: ExecutionResult, booking: dict) -> AgentReply:
        state.last_outcome = result.model_dump(include={"status", "reason", "action"})
        verb = {"cancel": "Cancelled", "reschedule": "Rescheduled"}[result.action]
        if result.ok:
            b = result.booking or booking
            state.transition(Phase.DONE, run.tracer, reason=f"{result.action} verified")
            state.last_completed_approval_id = result.approval_id
            when = f" to {human_slot(b['slot'])}" if result.action == "reschedule" else ""
            return self._finish(run, state, f"{verb} and verified: {b['provider_name']} ({b['service']}), booking "
                                f"{b['booking_id']}{when}.", outcome=result)
        if result.status == "unknown":
            return self._finish(run, state, f"I sent the {result.action} request but couldn't verify it yet. I'll "
                                "keep checking and won't send it twice.", outcome=result)
        if result.reason == "slot_conflict":
            state.transition(Phase.PRESENTING, run.tracer, reason="slot conflict")
            alts = self._reschedule_alternatives(run, booking, result)
            return self._finish(run, state, "That slot conflicts with the provider's schedule, so your booking "
                                f"is unchanged. {alts}", outcome=result)
        state.transition(Phase.FAILED, run.tracer, reason=result.reason or "failed")
        return self._finish(run, state, f"I couldn't {result.action} it ({(result.reason or '').replace('_', ' ')}). "
                            "I checked, and your booking is unchanged. Want me to try again?", outcome=result)

    # ------------------------------------------------------------- helpers
    def _alternatives(self, state: WorkflowState, exclude_slot: str) -> str:
        opts = []
        for c in state.candidates:
            for s in c["matching_slots"]:
                if s != exclude_slot:
                    opts.append(f"{c['name']} {human_slot(s)} (₹{c['price']})")
                    break
            if len(opts) == 3:
                break
        return ("Other options: " + "; ".join(opts) + ". Which would you like?") if opts else \
            "Shall I search again?"

    def _reschedule_alternatives(self, run: Run, booking: dict, result: ExecutionResult) -> str:
        try:
            details = run.gateway.call("get_provider", provider_id=booking["provider_id"])
            target = parse_dt(run.approvals.get(result.approval_id).details.slot)  # type: ignore[union-attr]
            day = target.date()
            res = run.gateway.call("search_providers", service=booking["service"], area=details["area"],
                                   date_range={"start": day.isoformat(),
                                               "end": (day + timedelta(days=1)).isoformat()})
            slots = next((p["available_slots"] for p in res["providers"]
                          if p["provider_id"] == booking["provider_id"]), [])
            slots = [s for s in slots if s != fmt_dt(target)][:3]
        except (ToolError, BudgetExceeded):
            slots = []
        if not slots:
            return "I couldn't find another free slot nearby. Want me to look at other days?"
        return "Free slots with the same provider: " + ", ".join(human_slot(s) for s in slots) + \
            ". Which one should I request?"

    def _where_we_are(self, state: WorkflowState) -> str:
        if state.phase == Phase.EXECUTING:
            return "An approved action is still being verified; I'll report its result."
        if state.pending_approval_id:
            return "A confirmation is still waiting for you."
        if state.hold:
            return f"You still have {state.hold.provider_name} on hold for {human_slot(state.hold.slot)}."
        if state.candidates:
            return "Your shortlist is saved. Tell me which option you'd like."
        return "Your request details are saved. Just tell me how you'd like to continue."

    def _limit_message(self, state: WorkflowState, exc: BudgetExceeded) -> str:
        what = {"model_turns": "the step limit", "tool_calls": "the tool-call limit",
                "wall_clock_seconds": "the time limit"}.get(exc.kind, "a limit")
        return f"I hit {what} for a single request, so I stopped here. Nothing was booked or changed. " + \
            self._where_we_are(state)

    def _guard_claims(self, run: Run, text: str | None) -> str:
        """The model never executes side effects, so any first-person claim that it did is false."""
        if not text or not _CLAIM_RE.search(text):
            return text or ""
        run.tracer.emit("guard", rule="unverified_success_claim", original=text)
        kept = [s for s in re.split(r"(?<=[.!?])\s+", text) if not _CLAIM_RE.search(s)]
        return ("Nothing has been booked or changed yet. That only happens after you confirm, and I'll report the "
                "verified result. " + " ".join(kept)).strip()

    # ----------------------------------------------------------- proactive
    def poll_events(self) -> list[AgentReply]:
        """Run periodically, without user input."""
        replies = self._reconcile_in_flight(kind="reconcile")
        run = self._run("provider_watch", quiet=True)  # invisible unless it finds something
        try:
            bookings = run.gateway.call("list_bookings", user_id=self.user_id)["bookings"]
        except (ToolError, BudgetExceeded) as exc:
            run.tracer.promote()
            run.tracer.emit("error", source="watch", error=str(exc))
            return replies
        known = {r["booking_id"]: r["status"] for r in self.store.db.execute("SELECT * FROM booking_watch")}
        for b in bookings:
            # Any transition INTO cancelled_by_provider counts, including bookings created
            # since the last poll (unknown to the snapshot). See BUGS.md #2.
            if b["status"] == "cancelled_by_provider" and known.get(b["booking_id"]) != "cancelled_by_provider":
                run.tracer.promote()
                run.tracer.emit("event_detected", event="provider_cancelled", booking_id=b["booking_id"])
                reply = self._handle_provider_cancel(run, b)
                if reply:
                    replies.append(reply)
        self._snapshot(bookings)
        return replies

    def _handle_provider_cancel(self, run: Run, booking: dict) -> AgentReply | None:
        slot = parse_dt(booking["slot"])
        if slot <= self.clock.now():
            run.tracer.emit("decision", what="no_notification", reason="booking already in the past")
            return None
        hours = (slot - self.clock.now()).total_seconds() / 3600
        run.tracer.emit("decision", what="notify_and_find_replacement", urgency="high" if hours < 24 else "normal")
        state = self.workflows.new(self.user_id, origin="proactive")
        toolbox = self._toolbox(run, state, "")
        area = run.gateway.call("get_provider", provider_id=booking["provider_id"])["area"]
        budget = self.memory.preference("budget_max", booking["service"])
        state.request = ErrandRequest(
            intent="book", service=booking["service"], area=area,
            date_start=slot.date().isoformat(), date_end=(slot.date() + timedelta(days=2)).isoformat(),
            budget_max=None if budget else int(math.ceil(booking["price"] * 1.25 / 10) * 10),
            sources={"service": f"booking:{booking['booking_id']}", "area": "provider_area",
                     "date_start": "original_slot", "date_end": "original_slot+2d"})
        lead = (f"Heads up: {booking['provider_name']} cancelled your {booking['service'].replace('_', ' ')} "
                f"booking for {human_slot(booking['slot'])}.")
        try:
            found = toolbox.t_search_providers(booking["service"], area,
                                               {"start": state.request.date_start, "end": state.request.date_end})
            options = [(c, s) for c in found["providers"] if c["provider_id"] != booking["provider_id"]
                       for s in c["matching_slots"]]
            if not options:
                text = lead + " I couldn't find a replacement that fits your preferences in the next 3 days. " \
                              "Should I widen the search?"
                return self._finish(run, toolbox.state, text, kind="notification")
            # best-ranked provider, slot closest to the original time
            best_provider = options[0][0]["provider_id"]
            choice = min((o for o in options if o[0]["provider_id"] == best_provider),
                         key=lambda o: abs((parse_dt(o[1]) - slot).total_seconds()))
            held = toolbox.t_hold_slot(choice[0]["provider_id"], choice[1])
            if "error" in held:
                raise ToolError(held["error"])
            out = toolbox.t_request_approval("book")
            if "error" in out:
                raise ToolError(out["error"])
        except (ToolError, BudgetExceeded) as exc:
            run.tracer.emit("error", source="replacement", error=str(exc))
            return self._finish(run, toolbox.state, lead + " I tried to line up a replacement but the provider "
                                "system had trouble. Want me to search again?", kind="notification")
        card = toolbox.approval_card
        text = lead + " I found a replacement within your preferences. Nothing is booked until you confirm.\n\n" + \
            self._card_text(card, toolbox.state)
        return self._finish(run, toolbox.state, text, card=card, kind="notification")

    def _reconcile_in_flight(self, kind: str) -> list[AgentReply]:
        pending = self.store.db.execute(
            "SELECT 1 FROM action_log WHERE status IN ('pending','unknown') LIMIT 1").fetchone()
        if not pending:
            return []
        run = self._run(kind)
        state = self.workflows.active(self.user_id)
        results = run.executor.recover()
        replies = []
        for r in results:
            if r.status == "unknown":
                continue
            if state.phase == Phase.EXECUTING:
                if r.action == "book":
                    replies.append(self._after_book(run, state, r, state.hold or Hold.model_validate(
                        run.approvals.get(r.approval_id).context["hold"]), rehold_attempted=True))  # type: ignore
                else:
                    replies.append(self._after_mutation(run, state, r, state.target_booking or {}))
                replies[-1].kind = "notification"
        if not replies:
            self.workflows.save(state)
            run.tracer.emit("run_end", phase=state.phase.value, reply=None, budget=run.budget.snapshot())
        return replies
