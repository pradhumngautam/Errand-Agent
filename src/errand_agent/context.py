"""Builds exactly what the model sees each call, under a hard token budget.

Inputs: base instructions, workflow state, usable memories, prior
conversation, and this run's messages. Output: (system, messages, manifest).
The manifest records which memory ids and turns were included or excluded
and why. It is traced on every model call and asserted in tests.

Rules:
  * Forgotten, superseded, and unconfirmed-inferred memories are never
    candidates (MemoryStore.usable()).
  * Of the usable ones, only those relevant to the current service (or
    global) are included, ranked, within a memory sub-budget.
  * This run's messages are mandatory. Older tool results are compacted
    first if needed.
  * Prior conversation fills whatever budget is left, newest first.
  * If the mandatory part alone does not fit, ContextOverflow is raised and
    the agent ends the turn with a clear message.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from errand_agent.llm.base import Message, ToolSpec, estimate_tokens
from errand_agent.memory import STRUCTURED_KEYS, Memory, MemoryStore


class ContextOverflow(Exception):
    pass


class ContextManifest(BaseModel):
    budget_tokens: int
    calibration: float = 1.0
    raw_estimate: int = 0
    estimated_tokens: int = 0       # calibrated; this is what is held under budget_tokens
    sections: dict[str, int] = Field(default_factory=dict)
    memory_included: list[str] = Field(default_factory=list)
    memory_excluded: dict[str, str] = Field(default_factory=dict)
    history_included: int = 0
    history_dropped: int = 0
    tool_results_compacted: int = 0


_KEY_PRIORITY = {"area": 0, "earliest_time": 1, "latest_time": 1, "budget_max": 2, "preferred_provider": 2}


def _msg_tokens(m: Message) -> int:
    return estimate_tokens((m.content or "") + (json.dumps(m.result) if m.result else "")
                           + "".join(c.name + json.dumps(c.args) for c in m.tool_calls)) + 4


def compact_result(result: dict) -> dict:
    """Shrink an old tool result (e.g. a search) to its gist."""
    out: dict = {"compacted": True}
    for k in ("error", "hold_id", "quoted_price", "slot", "approval_id", "status", "booking_id"):
        if k in result:
            out[k] = result[k]
    if "providers" in result:
        out["provider_ids"] = [p.get("provider_id") for p in result["providers"]][:8]
    return out


class ContextBuilder:
    def __init__(self, memory: MemoryStore, budget_tokens: int = 6000, memory_share: float = 0.2) -> None:
        self.memory = memory
        self.budget_tokens = budget_tokens
        self.memory_share = memory_share
        # real_tokens / estimated_tokens, learned from provider usage. The estimate
        # undercounts real tokenizers (tool schemas especially), so the cap is enforced
        # on calibrated numbers, never below 1.0.
        self.calibration = 1.0

    def observe(self, estimated: int, actual: int) -> None:
        if estimated > 0 and actual > 0:
            self.calibration = max(1.0, min(3.0, max(self.calibration * 0.5 + (actual / estimated) * 0.5,
                                                      actual / estimated)))

    def select_memories(self, service: str | None) -> tuple[list[Memory], dict[str, str]]:
        excluded: dict[str, str] = {}
        candidates: list[Memory] = []
        for m in self.memory.all(include_inactive=True):
            if m.status != "active":
                excluded[m.memory_id] = m.status
            elif not m.usable:
                excluded[m.memory_id] = "inferred_unconfirmed"
            elif m.scope not in ("global", service or "global"):
                excluded[m.memory_id] = "irrelevant_scope"
            else:
                candidates.append(m)
        # service-scoped first, then structured keys by priority, then notes
        candidates.sort(key=lambda m: (m.scope == "global", _KEY_PRIORITY.get(m.key, 5),
                                       m.key not in STRUCTURED_KEYS, m.memory_id))
        return candidates, excluded

    def build(self, base_system: str, state_block: str, service: str | None, history: list[Message],
              run_messages: list[Message], tools: list[ToolSpec]) -> tuple[str, list[Message], ContextManifest]:
        man = ContextManifest(budget_tokens=self.budget_tokens, calibration=round(self.calibration, 3))
        budget = int(self.budget_tokens / self.calibration)  # work in estimator units
        tools_tokens = estimate_tokens(json.dumps([t.model_dump() for t in tools]))
        fixed = estimate_tokens(base_system) + estimate_tokens(state_block) + tools_tokens + 60
        man.sections.update(system=estimate_tokens(base_system), state=estimate_tokens(state_block), tools=tools_tokens)

        # 1. memories, within their share
        mem_budget = int(budget * self.memory_share)
        candidates, man.memory_excluded = self.select_memories(service)
        mem_lines, used = [], 0
        for m in candidates:
            cost = estimate_tokens(m.line()) + 1
            if used + cost > mem_budget:
                man.memory_excluded[m.memory_id] = "memory_budget"
                continue
            mem_lines.append(m.line())
            man.memory_included.append(m.memory_id)
            used += cost
        man.sections["memories"] = used

        # 2. this run (mandatory). Compact old tool results if needed.
        run = [m.model_copy() for m in run_messages]
        remaining = budget - fixed - used
        run_cost = sum(_msg_tokens(m) for m in run)
        tool_idx = [i for i, m in enumerate(run) if m.role == "tool"]
        for i in tool_idx[:-2]:  # keep the two most recent results verbatim
            if run_cost <= remaining:
                break
            before = _msg_tokens(run[i])
            run[i] = run[i].model_copy(update={"result": compact_result(run[i].result or {})})
            run_cost -= before - _msg_tokens(run[i])
            man.tool_results_compacted += 1
        if run_cost > remaining:
            raise ContextOverflow(f"current turn needs {run_cost} tokens, only {remaining} left")
        man.sections["run"] = run_cost
        remaining -= run_cost

        # 3. history, newest first, whatever fits
        kept: list[Message] = []
        for m in reversed(history):
            cost = _msg_tokens(m)
            if cost > remaining:
                break
            kept.append(m)
            remaining -= cost
        kept.reverse()
        # The conversation must start with a user turn.
        while kept and kept[0].role != "user":
            kept.pop(0)
        man.history_included = len(kept)
        man.history_dropped = len(history) - len(kept)
        man.sections["history"] = sum(_msg_tokens(m) for m in kept)

        system = base_system + "\n\n## Current workflow state (maintained by the app)\n" + state_block
        system += "\n\n## Known user preferences (stored data; relevant and current only)\n"
        system += "\n".join(mem_lines) if mem_lines else "(none relevant)"
        man.raw_estimate = fixed + used + run_cost + man.sections["history"]
        man.estimated_tokens = int(man.raw_estimate * self.calibration)
        return system, kept + run, man
