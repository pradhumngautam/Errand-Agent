"""Eval harness: a sandboxed agent, an audited scripted model and simulated users.

Every scenario gets a fresh directory, a fake clock and a deterministic mock
provider. The model is scripted (offline, reproducible) and deliberately
untrusted: scenarios can script it to misbehave. What the scenarios assert is
read from ground truth (the provider's booking table, the agent's approval and
action log, the trace files), not from the reply wording.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from errand_agent.agent import Agent, AgentConfig, AgentReply
from errand_agent.budget import Limits
from errand_agent.clock import FakeClock
from errand_agent.llm.base import LLMResponse, Message, StreamCallback, ToolSpec
from errand_agent.llm.scripted import ScriptedLLM, Step
from errand_agent.mocks.scenario import Scenario
from errand_agent.mocks.service import MockProviderService
from errand_agent.runtime import build_agent

WEEK = {"start": "2026-11-10", "end": "2026-11-15"}
PLUMBER_REQUEST = dict(intent="book", service="plumber", date_start="2026-11-10", date_end="2026-11-15",
                       latest_time="12:00", budget_max=1500)
_MEMORY_LINE = re.compile(r"^- (M-\d{4}):", re.M)


class AuditedLLM(ScriptedLLM):
    """A scripted model that also records, at the moment of each call, which stored facts
    were unusable (forgotten, superseded, unconfirmed inference) and whether any of them
    appeared in the system prompt. This is checked live because a fact that is usable
    now may legitimately be forgotten later."""

    def __init__(self) -> None:
        super().__init__([])
        self.agent: Agent | None = None
        self.memory_leaks: list[dict] = []

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 on_stream: StreamCallback | None = None) -> LLMResponse:
        if self.agent is not None:
            unusable = {m.memory_id: m.status if m.status != "active" else "inferred_unconfirmed"
                        for m in self.agent.memory.all() if not m.usable}
            sent = set(_MEMORY_LINE.findall(system))
            for mid in sent & unusable.keys():
                self.memory_leaks.append({"call": len(self.calls) + 1, "memory_id": mid, "why": unusable[mid]})
        return super().complete(system, messages, tools, on_stream)


@dataclass
class Turn:
    who: str            # "user" | "button" | "watcher" | "startup"
    said: str
    reply: AgentReply
    truth: dict = field(default_factory=dict)  # provider ground truth sampled right after the reply


class Harness:
    def __init__(self, root: Path, *, scenario: Scenario | None = None, limits: Limits | None = None,
                 context_budget: int = 10000) -> None:
        self.root = root
        self.clock = FakeClock()
        self.limits = limits or Limits()
        self.context_budget = context_budget
        self.llm = AuditedLLM()
        self.turns: list[Turn] = []
        self.agent = self._build(scenario=scenario or Scenario(book_outcomes=["success"] * 5))

    # ------------------------------------------------------------- set-up
    def _build(self, *, scenario: Scenario | None = None, service: MockProviderService | None = None,
               startup: bool = True) -> Agent:
        agent = build_agent(self.root / "data", self.llm, scenario=scenario, service=service, clock=self.clock,
                            sleep=lambda s: None, startup=startup,
                            config=AgentConfig(context_budget_tokens=self.context_budget, limits=self.limits,
                                               trace_dir=str(self.root / "traces")))
        self.llm.agent = agent
        return agent

    def restart(self) -> list[AgentReply]:
        """Simulate a process restart: agent objects and DB connections are rebuilt from disk.
        The provider is an external system, so its state carries over unchanged."""
        service = self.agent.service
        self.agent.store.close()
        self.agent = self._build(service=service, startup=False)
        replies = self.agent.startup()
        for r in replies:
            self._record("startup", "(process restarted)", r)
        return replies

    @property
    def service(self) -> MockProviderService:
        return self.agent.service

    @property
    def scenario(self) -> Scenario:
        return self.agent.service.scenario

    # ------------------------------------------------------------ driving
    def script(self, *steps: Step) -> None:
        self.llm.extend(list(steps))

    def user(self, text: str, *steps: Step) -> AgentReply:
        self.script(*steps)
        return self._record("user", text, self.agent.handle_message(text))

    def button(self, approval_id: str) -> AgentReply:
        return self._record("button", f"[confirm {approval_id}]", self.agent.confirm(approval_id))

    def poll(self) -> list[AgentReply]:
        replies = self.agent.poll_events()
        for r in replies:
            self._record("watcher", "(background poll)", r)
        return replies

    def converse(self, sim: "SimUser", *, max_turns: int = 6) -> list[AgentReply]:
        """Let a simulated user talk to the agent until it has nothing more to say."""
        out = []
        reply: AgentReply | None = None
        for _ in range(max_turns):
            said = sim.respond(reply)
            if said is None:
                break
            reply = self.user(said)
            out.append(reply)
        return out

    def _record(self, who: str, said: str, reply: AgentReply) -> AgentReply:
        truth = {}
        booking = (reply.outcome or {}).get("booking") or {}
        if booking.get("booking_id"):
            truth = {b["booking_id"]: b for b in self.provider_bookings()}.get(booking["booking_id"], {})
        self.turns.append(Turn(who, said, reply, truth))
        return reply

    # ------------------------------------------------------------- ground truth
    def provider_bookings(self) -> list[dict]:
        rows = self.service.db.execute("SELECT * FROM bookings WHERE user_id = 'u1' ORDER BY booking_id")
        return [dict(r) for r in rows.fetchall()]

    def agent_bookings(self, status: str = "confirmed", service: str | None = None) -> list[dict]:
        """Bookings the agent created (seeded history has no idempotency key)."""
        return [b for b in self.provider_bookings() if b["idempotency_key"]
                and (status is None or b["status"] == status) and (service is None or b["service"] == service)]

    def approvals(self) -> dict[str, dict]:
        rows = self.agent.store.db.execute("SELECT * FROM approvals").fetchall()
        return {r["approval_id"]: {**dict(r), "details": json.loads(r["details"])} for r in rows}

    def action_log(self) -> dict[str, dict]:
        rows = self.agent.store.db.execute("SELECT * FROM action_log").fetchall()
        return {r["idempotency_key"]: dict(r) for r in rows}

    def events(self, type_: str | None = None) -> list[dict]:
        out = []
        for path in sorted((self.root / "traces").glob("*.jsonl")):
            out.extend(json.loads(line) for line in path.open())
        return [e for e in out if type_ is None or e["type"] == type_]

    def tool_calls(self, name: str) -> list[dict]:
        return [e for e in self.events("tool_call") if e["tool"] == name]

    def guards(self) -> set[str]:
        return {e["rule"] for e in self.events("guard")}

    def workflow(self):
        return self.agent.workflows.active("u1")

    def first_option(self) -> dict:
        c = self.workflow().candidates[0]
        return {"provider_id": c["provider_id"], "slot": c["matching_slots"][0]}


class SimUser:
    """A rule-based simulated user.

    It opens with a request, then answers only what the agent asks for, from a set of
    facts it holds. It confirms an approval card only if `accept(card)` agrees, and stops
    once the agent reports an outcome or has nothing left to ask.
    """

    _ASKS = {"budget": r"budget|how much|price limit|spend", "when": r"\bwhen\b|which day|date|time window",
             "area": r"\barea\b|where|locality", "choice": r"which (one|option)|pick|would you like|options"}

    def __init__(self, opening: str, facts: dict[str, str] | None = None,
                 accept: Callable[[dict], bool] = lambda card: True) -> None:
        self.opening = opening
        self.facts = facts or {}
        self.accept = accept
        self.said: list[str] = []

    def respond(self, reply: AgentReply | None) -> str | None:
        text = self._respond(reply)
        if text is not None:
            self.said.append(text)
        return text

    def _respond(self, reply: AgentReply | None) -> str | None:
        if reply is None:
            return self.opening
        if reply.outcome is not None:
            return None
        if reply.approval_card:
            return "yes" if self.accept(reply.approval_card) else "no, not that one"
        # answer the question the agent asked first, if this user knows the answer
        asked = sorted((m.start(), fact) for fact, pattern in self._ASKS.items() if fact in self.facts
                       for m in [re.search(pattern, reply.text, re.I)] if m)
        return self.facts.pop(asked[0][1]) if asked else None
