"""The tools the model can call, with their guards.

There is no book / cancel / reschedule tool. The most the model can do is
`request_approval`, which builds the approval from authoritative state (the
current hold, or the booking as returned by list_bookings), never from
model-supplied prices or slots.

Guards that do not trust the model:
  * cancel / reschedule intent must be grounded in the user's own words
  * an "explicit" preference must appear in the user's message, otherwise
    it is stored as inferred and unconfirmed
  * raw reviews never reach the model (see reviews.py)
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Callable

from errand_agent.approvals import ApprovalManager, MaterialDetails
from errand_agent.budget import BudgetExceeded
from errand_agent.clock import Clock, fmt_dt, parse_dt, slot_label
from errand_agent.executor import Hold
from errand_agent.gateway import ToolGateway
from errand_agent.llm.base import ToolSpec
from errand_agent.memory import MemoryStore
from errand_agent.mocks.catalog import AREAS, SERVICES
from errand_agent.mocks.errors import ToolError
from errand_agent.tracing import Tracer
from errand_agent.workflow import ErrandRequest, Phase, WorkflowState

_CANCEL_RE = re.compile(r"\b(cancel|call off|don'?t need|do not need|scrap|drop)\b", re.I)
_RESCHEDULE_RE = re.compile(r"\b(move|resched\w*|shift|change|postpone|prepone|push|instead|another (day|time)|"
                            r"different (day|time))\b", re.I)
_AFFIRM_RE = re.compile(r"\b(yes|yeah|yep|correct|right|confirm\w*|that'?s (true|right)|exactly)\b", re.I)
_SEARCH_FIELDS = {"service", "area", "date_start", "date_end", "earliest_time", "latest_time", "budget_max",
                  "no_budget_limit", "provider_id"}


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(name="update_request", description="Record details the user gave or corrected. Send only changed fields. "
             "Returns the full request and which required details are still missing.",
             parameters=_schema({
                 "intent": {"type": "string", "enum": ["book", "reschedule", "cancel"]},
                 "service": {"type": "string", "enum": list(SERVICES)},
                 "area": {"type": "string", "enum": list(AREAS)},
                 "date_start": {"type": "string", "description": "YYYY-MM-DD"},
                 "date_end": {"type": "string", "description": "YYYY-MM-DD"},
                 "earliest_time": {"type": "string", "description": "HH:MM 24h"},
                 "latest_time": {"type": "string", "description": "HH:MM 24h"},
                 "budget_max": {"type": "integer", "description": "max price in INR"},
                 "no_budget_limit": {"type": "boolean"},
                 "provider_id": {"type": "string", "description": "a specific provider the user wants"},
                 "booking_id": {"type": "string", "description": "existing booking to reschedule/cancel"},
                 "new_slot": {"type": "string", "description": "ISO datetime for reschedule"},
                 "clear_fields": {"type": "array", "items": {"type": "string"}},
             })),
    ToolSpec(name="search_providers", description="Search providers; results are filtered to the request's budget "
             "and time window and ranked.",
             parameters=_schema({
                 "service": {"type": "string", "enum": list(SERVICES)},
                 "area": {"type": "string", "enum": list(AREAS)},
                 "date_range": _schema({"start": {"type": "string"}, "end": {"type": "string"}}, ["start", "end"]),
             }, ["service", "area", "date_range"])),
    ToolSpec(name="get_provider", description="Provider details plus a pre-screened review digest.",
             parameters=_schema({"provider_id": {"type": "string"}}, ["provider_id"])),
    ToolSpec(name="list_bookings", description="The user's past and upcoming bookings.", parameters=_schema({})),
    ToolSpec(name="hold_slot", description="Hold a provider's slot for 10 minutes and get the quoted price.",
             parameters=_schema({"provider_id": {"type": "string"},
                                 "slot": {"type": "string", "description": "ISO datetime from available slots"}},
                                ["provider_id", "slot"])),
    ToolSpec(name="request_approval", description="Show the user a confirmation card for the action. For book, "
             "uses the current hold. The app executes only after the user confirms.",
             parameters=_schema({"action": {"type": "string", "enum": ["book", "cancel", "reschedule"]},
                                 "booking_id": {"type": "string"}, "new_slot": {"type": "string"}}, ["action"])),
    ToolSpec(name="remember_preference", description="Store a lasting preference the user explicitly stated. "
             "Keys: area, earliest_time, latest_time, budget_max, preferred_provider, or a short note_* key.",
             parameters=_schema({"key": {"type": "string"}, "value": {"type": "string"},
                                 "scope": {"type": "string", "description": "'global' or a service name"}},
                                ["key", "value"])),
    ToolSpec(name="forget_preference", description="Forget stored facts by memory_id or key.",
             parameters=_schema({"memory_id": {"type": "string"}, "key": {"type": "string"},
                                 "scope": {"type": "string"}})),
    ToolSpec(name="list_memories", description="Show what is stored about the user, with provenance.",
             parameters=_schema({"include_history": {"type": "boolean"}})),
    ToolSpec(name="confirm_memory", description="Mark an inferred fact as confirmed after the user agrees.",
             parameters=_schema({"memory_id": {"type": "string"}}, ["memory_id"])),
]


def _err(code: str, message: str = "", **extra: Any) -> dict:
    return {"error": code, "message": message or code, **extra}


def slot_fits(slot: str, req: ErrandRequest) -> bool:
    dt = parse_dt(slot)
    hhmm = dt.strftime("%H:%M")
    if req.date_start and dt.date() < date.fromisoformat(req.date_start):
        return False
    if req.date_end and dt.date() > date.fromisoformat(req.date_end):
        return False
    if req.earliest_time and hhmm < req.earliest_time:
        return False
    if req.latest_time and hhmm > req.latest_time:
        return False
    return True


class AgentToolbox:
    def __init__(self, *, state: WorkflowState, gateway: ToolGateway, approvals: ApprovalManager,
                 memory: MemoryStore, analyzer: Any, clock: Clock, tracer: Tracer, user_id: str,
                 user_text: str, new_workflow: Callable[[], WorkflowState]) -> None:
        self.state = state
        self.gateway = gateway
        self.approvals = approvals
        self.memory = memory
        self.analyzer = analyzer
        self.clock = clock
        self.tracer = tracer
        self.user_id = user_id
        self.user_text = user_text
        self.new_workflow = new_workflow
        self.approval_card: dict | None = None

    # ------------------------------------------------------------ dispatch
    def dispatch(self, name: str, args: dict) -> dict:
        handler = getattr(self, f"t_{name}", None)
        if handler is None:
            self.tracer.emit("guard", rule="unknown_tool", tool=name)
            return _err("unknown_tool", f"'{name}' is not available. Bookings, cancellations and reschedules "
                        "happen only after the user confirms an approval card.")
        try:
            return handler(**(args or {}))
        except TypeError as exc:
            return _err("bad_arguments", str(exc))
        except BudgetExceeded:
            raise
        except ToolError as exc:
            return _err(exc.code, exc.message, retryable=exc.retryable)

    @property
    def eff(self) -> ErrandRequest:
        return self.state.request.effective(self.memory)

    # ------------------------------------------------------------- request
    def t_update_request(self, clear_fields: list[str] | None = None, **fields: Any) -> dict:
        if self.state.phase == Phase.EXECUTING:
            return _err("action_in_progress", "an approved action is still being verified; wait for its result")
        if self.state.phase in (Phase.DONE, Phase.FAILED) and fields:
            self.state = self.new_workflow()  # the agent reads toolbox.state back after the loop
        known = set(ErrandRequest.model_fields) - {"sources"}
        unknown = set(fields) - known
        if unknown:
            return _err("unknown_fields", ", ".join(sorted(unknown)))
        intent = fields.get("intent")
        if intent == "cancel" and not _CANCEL_RE.search(self.user_text):
            self.tracer.emit("guard", rule="ungrounded_intent", intent=intent)
            return _err("intent_not_grounded", "The user has not asked to cancel anything in their own words.")
        if intent == "reschedule" and not _RESCHEDULE_RE.search(self.user_text):
            self.tracer.emit("guard", rule="ungrounded_intent", intent=intent)
            return _err("intent_not_grounded", "The user has not asked to reschedule in their own words.")

        before = self.state.request.model_copy(deep=True)
        updated = before.model_copy(deep=True)
        for f in clear_fields or []:
            if f in known:
                setattr(updated, f, ErrandRequest.model_fields[f].default)
                updated.sources.pop(f, None)
        for f, v in fields.items():
            if v is None:
                continue
            setattr(updated, f, v)
            updated.sources[f] = "user"
        if fields.get("budget_max") is not None:
            updated.no_budget_limit = False
        problems = updated.validate_fields()
        if problems:
            return _err("invalid_fields", "; ".join(problems))
        self.state.request = updated
        changed = {f for f in known if getattr(before, f) != getattr(updated, f)}
        if changed & _SEARCH_FIELDS:
            self.state.candidates = []
        eff = self.eff
        if self.state.phase != Phase.COLLECTING or changed:
            self.state.transition(Phase.COLLECTING, self.tracer, reason=f"request updated: {sorted(changed)}")
        return {"request": eff.model_dump(exclude_none=True), "changed": sorted(changed), "missing": eff.missing()}

    # -------------------------------------------------------------- search
    def t_search_providers(self, service: str, area: str, date_range: dict) -> dict:
        result = self.gateway.call("search_providers", service=service, area=area, date_range=date_range)
        eff = self.eff
        preferred = eff.provider_id
        if not preferred:
            mem = self.memory.preference("preferred_provider", service)
            preferred = mem.value if mem else None
        constraints = eff.model_copy(update={"date_start": date_range["start"][:10], "date_end": date_range["end"][:10]})
        ranked, excluded = [], {"over_budget": 0, "no_matching_slots": 0}
        for p in result["providers"]:
            if eff.budget_max is not None and p["price"] > eff.budget_max:
                excluded["over_budget"] += 1
                continue
            slots = [s for s in p["available_slots"] if slot_fits(s, constraints)]
            if not slots:
                excluded["no_matching_slots"] += 1
                continue
            score = p["rating"] * 20 - p["price"] / 100 + (5 if p["attributes"]["verified"] else 0)
            score += 50 if p["provider_id"] == preferred else 0
            ranked.append((score, {
                "provider_id": p["provider_id"], "name": p["name"], "price": p["price"], "rating": p["rating"],
                "reviews": p["review_count"], "verified": p["attributes"]["verified"],
                "warranty_days": p["attributes"]["warranty_days"], "preferred": p["provider_id"] == preferred,
                "matching_slots": slots[:6], "slot_labels": [slot_label(s) for s in slots[:6]],
                "service": service, "area": area,
            }))
        ranked.sort(key=lambda x: -x[0])
        self.state.candidates = [c for _, c in ranked][:5]
        self.state.transition(Phase.PRESENTING, self.tracer, reason="search results")
        return {"providers": self.state.candidates, "excluded": excluded,
                "constraints_applied": {k: getattr(eff, k) for k in ("budget_max", "earliest_time", "latest_time")}}

    def t_get_provider(self, provider_id: str) -> dict:
        details = self.gateway.call("get_provider", provider_id=provider_id)
        reviews = details.pop("reviews", [])
        digest = self.analyzer.analyze(reviews)
        if digest.get("injection_suspected"):
            self.tracer.emit("untrusted_content", provider_id=provider_id, flagged_reviews=digest["flagged_reviews"],
                             action="excluded_from_model_context")
        details["review_digest"] = digest
        return details

    def t_list_bookings(self) -> dict:
        data = self.gateway.call("list_bookings", user_id=self.user_id)
        now = self.clock.now()
        for b in data["bookings"]:
            b["when"] = "upcoming" if parse_dt(b["slot"]) > now else "past"
            b["slot_label"] = slot_label(b["slot"])
        return {"bookings": data["bookings"]}

    # ---------------------------------------------------------------- hold
    def _provider_meta(self, provider_id: str) -> dict:
        for c in self.state.candidates:
            if c["provider_id"] == provider_id:
                return {"name": c["name"], "service": c["service"], "price": c["price"]}
        details = self.gateway.call("get_provider", provider_id=provider_id)
        return {"name": details["name"], "service": details["service"], "price": details["price"]}

    def t_hold_slot(self, provider_id: str, slot: str) -> dict:
        try:
            parse_dt(slot)
        except ValueError:
            return _err("invalid_slot", "slot must be an ISO datetime from available slots")
        meta = self._provider_meta(provider_id)
        h = self.gateway.call("hold_slot", provider_id=provider_id, slot=slot)
        self.state.hold = Hold(hold_id=h["hold_id"], provider_id=provider_id, provider_name=meta["name"],
                               service=meta["service"], slot=h["slot"], quoted_price=h["quoted_price"],
                               expires_at=h["expires_at"])
        self.state.transition(Phase.HOLDING, self.tracer, reason=f"held {h['hold_id']}")
        out = {**h, "slot_label": slot_label(h["slot"]), "provider_name": meta["name"]}
        if h["quoted_price"] != meta["price"]:
            out["price_changed"] = {"search_price": meta["price"], "quoted_price": h["quoted_price"]}
        eff = self.eff
        if eff.budget_max is not None and h["quoted_price"] > eff.budget_max:
            out["over_budget"] = {"budget_max": eff.budget_max}
        if not slot_fits(h["slot"], eff.model_copy(update={"date_start": None, "date_end": None})):
            out["outside_time_window"] = True
        return out

    # ------------------------------------------------------------ approval
    def t_request_approval(self, action: str, booking_id: str | None = None, new_slot: str | None = None) -> dict:
        st = self.state
        if action == "book":
            if st.request.intent not in (None, "book"):
                return _err("intent_mismatch", f"current intent is {st.request.intent}")
            if st.hold is None:
                return _err("no_active_hold", "hold a slot first")
            if self.clock.now() >= datetime.fromisoformat(st.hold.expires_at):
                return _err("hold_expired", "the hold expired; hold the slot again")
            details = st.hold.details()
            context = {"hold": st.hold.model_dump()}
        elif action in ("cancel", "reschedule"):
            if st.request.intent != action:
                self.tracer.emit("guard", rule="approval_intent_mismatch", action=action, intent=st.request.intent)
                return _err("intent_mismatch", f"the user has not asked to {action}")
            booking_id = booking_id or st.request.booking_id
            if not booking_id:
                return _err("missing_booking_id")
            bookings = self.gateway.call("list_bookings", user_id=self.user_id)["bookings"]
            booking = next((b for b in bookings if b["booking_id"] == booking_id), None)
            if booking is None or booking["status"] != "confirmed":
                return _err("booking_not_actionable", "no confirmed booking with that id")
            slot = booking["slot"]
            if action == "reschedule":
                new_slot = new_slot or st.request.new_slot
                if not new_slot:
                    return _err("missing_new_slot")
                try:
                    slot = fmt_dt(parse_dt(new_slot))
                except ValueError:
                    return _err("invalid_slot")
            details = MaterialDetails(action=action, provider_id=booking["provider_id"],  # type: ignore[arg-type]
                                      provider_name=booking["provider_name"], service=booking["service"],
                                      slot=slot, price=booking["price"], booking_id=booking_id)
            st.target_booking = booking
            context = {"booking": booking, "new_slot": slot if action == "reschedule" else None}
        else:
            return _err("bad_action")
        approval = self.approvals.request(st.workflow_id, details, context=context)
        st.pending_approval_id = approval.approval_id
        st.transition(Phase.AWAITING_APPROVAL, self.tracer, reason=f"approval {approval.approval_id}")
        self.approval_card = {"approval_id": approval.approval_id, "action": action, "summary": details.summary(),
                              "details": details.model_dump(), "fingerprint": approval.fingerprint}
        return {"approval_id": approval.approval_id, "status": "awaiting_user_confirmation",
                "summary": details.summary()}

    # -------------------------------------------------------------- memory
    def _stated_by_user(self, value: str) -> bool:
        """Did the user's own message contain this value? Numbers are compared as numbers
        ("10am" matches "10:00", "₹1,500" matches "1500", "6pm" matches "18:00"). See BUGS.md #3."""
        text = self.user_text.lower()
        v = str(value).lower().strip()
        if v and v in text:
            return True
        plain = re.sub(r"(?<=\d),(?=\d)", "", text)
        text_nums = {int(n) for n in re.findall(r"\d+", plain)}
        text_nums |= {int(h) + 12 for h in re.findall(r"\b(\d{1,2})(?::\d{2})?\s*pm\b", plain) if int(h) < 12}
        value_nums = [int(n) for n in re.findall(r"\d+", v.replace(",", "")) if int(n) != 0]
        if value_nums:
            return all(n in text_nums for n in value_nums)
        words = re.findall(r"[a-z]{4,}", v)
        return bool(words) and all(w in text for w in words)

    def t_remember_preference(self, key: str, value: str, scope: str = "global") -> dict:
        if scope not in ("global", *SERVICES):
            return _err("bad_scope", f"scope must be global or one of {list(SERVICES)}")
        if self._stated_by_user(value):
            m = self.memory.remember(key, value, source="explicit", scope=scope, evidence=self.user_text[:200])
            return {"stored": m.model_dump(), "note": "explicit preference saved"}
        m = self.memory.remember(key, value, source="inferred", scope=scope, evidence=self.user_text[:200])
        self.tracer.emit("guard", rule="unstated_preference_downgraded", key=key)
        if m.value != str(value) or m.source == "explicit":
            return {"not_stored": True, "kept": m.model_dump(), "note": "the user did not state this value and it "
                    "conflicts with an explicit preference, so nothing changed"}
        return {"stored": m.model_dump(), "note": "user did not state this value, so it is saved as an UNCONFIRMED "
                "inference and will not be used until they confirm it"}

    def t_forget_preference(self, memory_id: str | None = None, key: str | None = None, scope: str | None = None) -> dict:
        ids = self.memory.forget(memory_id=memory_id, key=key, scope=scope)
        return {"forgotten": ids} if ids else _err("not_found", "no active memory matched")

    def t_list_memories(self, include_history: bool = False) -> dict:
        items = self.memory.all(include_inactive=include_history)
        return {"memories": [{"memory_id": m.memory_id, "key": m.key, "value": m.value, "scope": m.scope,
                              "source": m.source, "confirmed": m.confirmed, "status": m.status} for m in items]}

    def t_confirm_memory(self, memory_id: str) -> dict:
        if not _AFFIRM_RE.search(self.user_text):
            self.tracer.emit("guard", rule="memory_confirm_not_grounded", memory_id=memory_id)
            return _err("not_confirmed_by_user", "ask the user to confirm this fact first")
        m = self.memory.confirm(memory_id)
        return {"confirmed": m.model_dump()} if m else _err("not_found")
