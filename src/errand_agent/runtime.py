"""Wire the agent, mocks and storage together (used by the CLI, web app and evals)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from errand_agent.agent import Agent, AgentConfig
from errand_agent.clock import Clock, SystemClock
from errand_agent.llm.base import LLMClient
from errand_agent.mocks.scenario import Scenario
from errand_agent.mocks.service import MockProviderService
from errand_agent.seeds import seed_conversation, seed_memories
from errand_agent.store import AgentStore


def build_agent(data_dir: str | Path, llm: LLMClient, *, scenario: Scenario | None = None,
                clock: Clock | None = None, config: AgentConfig | None = None, seed: bool = True,
                service: MockProviderService | None = None, analyzer: Any = None,
                sleep: Callable[[float], None] | None = None, startup: bool = True,
                trace_listener: Callable[[dict], None] | None = None) -> Agent:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    clock = clock or SystemClock()
    service = service or MockProviderService(data_dir / "provider.db", clock, scenario or Scenario())
    store = AgentStore(data_dir / "agent.db")
    fresh = not store.db.execute("SELECT 1 FROM counters WHERE name = 'memory'").fetchone()
    kwargs: dict = {}
    if sleep is not None:
        kwargs["sleep"] = sleep
    agent = Agent(store=store, service=service, llm=llm, clock=clock, config=config or AgentConfig(),
                  analyzer=analyzer, trace_listener=trace_listener, **kwargs)
    if seed and fresh:
        seed_memories(agent.memory, clock)
        seed_conversation(store, clock, agent.user_id)
    if startup:
        agent.startup()
    return agent
