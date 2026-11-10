from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from errand_agent.approvals import ApprovalManager
from errand_agent.budget import Budget, Limits
from errand_agent.clock import FakeClock
from errand_agent.executor import ActionExecutor, Hold
from errand_agent.gateway import ToolGateway
from errand_agent.mocks.scenario import Scenario
from errand_agent.mocks.service import MockProviderService
from errand_agent.store import AgentStore
from errand_agent.tracing import Tracer


@dataclass
class Env:
    tmp: Path
    clock: FakeClock
    scenario: Scenario
    service: MockProviderService
    store: AgentStore
    tracer: Tracer
    budget: Budget
    gateway: ToolGateway
    approvals: ApprovalManager
    executor: ActionExecutor

    def restart(self) -> "Env":
        """Simulate an agent process restart: new connections and objects, same files.
        The provider side (mock service DB) is an external system and keeps its state."""
        self.store.close()
        return build_env(self.tmp, self.clock, self.scenario, service=self.service)

    def hold(self, provider_id: str = "PLU-KOR-01", day_offset: int = 1) -> Hold:
        day = self.clock.now().date().fromordinal(self.clock.now().date().toordinal() + day_offset)
        result = self.service.search_providers("plumber" if provider_id.startswith("PLU") else "electrician",
                                               "Koramangala", {"start": day.isoformat(), "end": day.isoformat()})
        provider = next(p for p in result["providers"] if p["provider_id"] == provider_id)
        assert provider["available_slots"], "fixture provider needs a slot"
        h = self.gateway.call("hold_slot", provider_id=provider_id, slot=provider["available_slots"][0])
        return Hold(hold_id=h["hold_id"], provider_id=provider_id, provider_name=provider["name"],
                    service=provider["service"], slot=h["slot"], quoted_price=h["quoted_price"],
                    expires_at=h["expires_at"])

    def approved(self, hold: Hold, workflow_id: str = "wf-1") -> str:
        a = self.approvals.request(workflow_id, hold.details(), context={"hold_id": hold.hold_id})
        self.approvals.approve(a.approval_id, via="button")
        return a.approval_id

    def bookings_for_hold_slot(self, hold: Hold) -> list[dict]:
        return [b for b in self.service.list_bookings("u1")["bookings"]
                if b["provider_id"] == hold.provider_id and b["slot"] == hold.slot and b["status"] == "confirmed"]

    def tool_calls(self, name: str) -> list[dict]:
        return [e for e in self.tracer.events if e["type"] == "tool_call" and e["tool"] == name]


def build_env(tmp: Path, clock: FakeClock, scenario: Scenario, service: MockProviderService | None = None,
              limits: Limits | None = None) -> Env:
    service = service or MockProviderService(tmp / "provider.db", clock, scenario)
    store = AgentStore(tmp / "agent.db")
    tracer = Tracer("test-run")
    budget = Budget(limits)
    gateway = ToolGateway(service, tracer, budget, sleep=lambda s: None)
    approvals = ApprovalManager(store, clock, tracer)
    executor = ActionExecutor(store, approvals, gateway, clock, tracer, limits)
    return Env(tmp, clock, scenario, service, store, tracer, budget, gateway, approvals, executor)


@pytest.fixture
def make_env(tmp_path: Path):
    def _make(scenario: Scenario | None = None, limits: Limits | None = None) -> Env:
        return build_env(tmp_path, FakeClock(), scenario or Scenario(), limits=limits)
    return _make
