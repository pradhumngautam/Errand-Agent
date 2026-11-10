"""Deterministic model stand-ins for offline evals.

A ScriptedLLM plays a list of steps. Each step is either a fixed LLMResponse
or a function of the conversation so far (so a script can say "hold the
first slot the last search returned" without hard-coding it).

The scripted model is deliberately *not* trusted: scenarios can script it to
fall for a prompt injection or call tools it should not, to prove that
application code, not model behaviour, enforces the invariants.
"""

from __future__ import annotations

import itertools
from typing import Callable, Union

from errand_agent.llm.base import LLMResponse, Message, StreamCallback, ToolCall, ToolSpec, Usage, estimate_tokens

_ids = itertools.count(1)


class ScriptView:
    """What a dynamic script step can read."""

    def __init__(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> None:
        self.system = system
        self.messages = messages
        self.tools = tools

    def last_result(self, tool: str | None = None) -> dict:
        for m in reversed(self.messages):
            if m.role == "tool" and (tool is None or m.name == tool):
                return m.result or {}
        raise LookupError(f"no result for {tool}")

    def last_user(self) -> str:
        return next((m.content or "" for m in reversed(self.messages) if m.role == "user"), "")


Step = Union[LLMResponse, Callable[[ScriptView], LLMResponse]]


def say(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="scripted")


def call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"call_{next(_ids)}", name=name, args=args)], model="scripted")


def calls(*pairs: tuple[str, dict]) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"call_{next(_ids)}", name=n, args=a) for n, a in pairs],
                       model="scripted")


def dynamic(fn: Callable[[ScriptView], LLMResponse]) -> Callable[[ScriptView], LLMResponse]:
    return fn


class ScriptExhausted(Exception):
    pass


class ScriptedLLM:
    model = "scripted"

    def __init__(self, steps: list[Step] | None = None, loop_last: bool = False) -> None:
        self.steps: list[Step] = list(steps or [])
        self.loop_last = loop_last
        self.calls: list[dict] = []  # what the model was sent, for context assertions

    def extend(self, steps: list[Step]) -> None:
        self.steps.extend(steps)

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 on_stream: StreamCallback | None = None) -> LLMResponse:
        self.calls.append({"system": system, "messages": [m.model_dump() for m in messages],
                           "tools": [t.name for t in tools]})
        if not self.steps:
            raise ScriptExhausted("scripted model has no more steps")
        step = self.steps[0] if (self.loop_last and len(self.steps) == 1) else self.steps.pop(0)
        resp = step(ScriptView(system, messages, tools)) if callable(step) else step.model_copy(deep=True)
        prompt = system + "".join((m.content or "") + str(m.result or "") + str(m.tool_calls) for m in messages)
        if on_stream and resp.text:  # mimic streaming so the UI path is testable offline
            for word in resp.text.split(" "):
                on_stream("text", word + " ")
        resp.usage = Usage(input_tokens=estimate_tokens(prompt),
                           output_tokens=estimate_tokens((resp.text or "") + str([c.args for c in resp.tool_calls])))
        return resp
