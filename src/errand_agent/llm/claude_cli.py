"""Use the locally installed Claude Code CLI (`claude -p`) as the model, with no API key.

Each model call runs one non-interactive `claude -p` process:
  * our system prompt replaces Claude Code's (`--system-prompt`)
  * Claude Code's built-in tools, MCP servers, settings and slash commands are switched off
  * the app's tools are described in the prompt as *actions*. The model requests them
    through a JSON schema (`--json-schema`), and the app parses them into ordinary ToolCalls.
So the agent loop, approvals, guards and traces are exactly the same as with an API
provider. Only the transport differs.

Config (env): CLAUDE_CLI_PATH (default: `claude` on PATH), CLAUDE_CLI_MODEL (default: haiku),
CLAUDE_CLI_TIMEOUT seconds (default 120).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid

from errand_agent.llm.base import LLMError, LLMResponse, Message, StreamCallback, ToolCall, ToolSpec, Usage

_ALIASES = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}

# Built-in runtime tools: disallowed so the CLI behaves as a plain model, not an agent.
_BUILTIN_TOOLS = ("Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task,Agent,NotebookEdit,TodoWrite,"
                  "MultiEdit,LS,BashOutput,KillShell,SlashCommand,Skill,ExitPlanMode")

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Reply to the user, or a short note when requesting actions"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args_json": {"type": "string", "description": "JSON object with the action's arguments"},
                },
                "required": ["name", "args_json"],
            },
        },
    },
    "required": ["text", "tool_calls"],
}

PROTOCOL = """

## How you act (output protocol)
You cannot call functions directly. You respond ONLY with the structured output:
- To use app actions, list them in `tool_calls` (name + args_json, a JSON object string). The app runs
  them and shows you each result as "ACTION RESULT" in the next turn. Keep `text` to a short note or empty.
- To reply to the user, put the reply (Markdown allowed) in `text` and leave `tool_calls` empty.
- Request actions only from the catalog below, with arguments matching their schemas.
"""


def _render_tools(tools: list[ToolSpec]) -> str:
    lines = ["\n## Action catalog"]
    for t in tools:
        lines.append(f"- {t.name}: {t.description}\n  args schema: {json.dumps(t.parameters, separators=(',', ':'))}")
    return "\n".join(lines)


def _render_transcript(messages: list[Message]) -> str:
    out = []
    for m in messages:
        if m.role == "user":
            out.append(f"USER: {m.content}")
        elif m.role == "assistant":
            if m.tool_calls:
                calls = "; ".join(f"{c.name}({json.dumps(c.args, ensure_ascii=False)})" for c in m.tool_calls)
                out.append(f"ASSISTANT requested actions: {calls}" + (f"  (note: {m.content})" if m.content else ""))
            else:
                out.append(f"ASSISTANT: {m.content}")
        else:
            out.append(f"ACTION RESULT {m.name}: {json.dumps(m.result, ensure_ascii=False)}")
    out.append("\nDecide your next step now (request actions, or reply to the user).")
    return "\n\n".join(out)


class ClaudeCLIClient:
    def __init__(self, model: str | None = None, cli_path: str | None = None, timeout: float | None = None) -> None:
        self.cli = cli_path or os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude") \
            or os.path.expanduser("~/.local/bin/claude")
        if not os.path.exists(self.cli):
            raise LLMError("local model runtime not found (set CLAUDE_CLI_PATH)")
        self.cli_model = model or os.environ.get("CLAUDE_CLI_MODEL") or "haiku"
        self.model = _ALIASES.get(self.cli_model, self.cli_model)
        self.calibrate_tokens = False  # reported usage includes runtime overhead; don't calibrate on it
        self.timeout = timeout or float(os.environ.get("CLAUDE_CLI_TIMEOUT", "120"))
        # empty working dir: no project files or CLAUDE.md are discovered
        self.workdir = tempfile.mkdtemp(prefix="errand-claude-cli-")

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec],
                 on_stream: StreamCallback | None = None) -> LLMResponse:
        cmd = [self.cli, "-p", "--output-format", "json", "--model", self.cli_model,
               "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
               "--disable-slash-commands", "--disallowedTools", _BUILTIN_TOOLS,
               "--json-schema", json.dumps(OUTPUT_SCHEMA),
               "--system-prompt", system + PROTOCOL + (_render_tools(tools) if tools else "")]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
        last_err = ""
        for attempt in range(2):
            try:
                proc = subprocess.run(cmd, input=_render_transcript(messages), capture_output=True, text=True,
                                      timeout=self.timeout, cwd=self.workdir, env=env)
            except subprocess.TimeoutExpired as exc:
                raise LLMError(f"model call timed out after {self.timeout:.0f}s") from exc
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                last_err = (proc.stderr or proc.stdout)[:300]
                time.sleep(1.5)
                continue
            if data.get("is_error") or proc.returncode != 0:
                last_err = str(data.get("result") or data.get("subtype") or proc.stderr)[:300]
                time.sleep(1.5)
                continue
            return self._parse(data, on_stream)
        raise LLMError(f"model call failed: {last_err}")

    def _parse(self, data: dict, on_stream: StreamCallback | None) -> LLMResponse:
        out = data.get("structured_output")
        if not isinstance(out, dict):
            try:
                out = json.loads(data.get("result") or "{}")
            except json.JSONDecodeError:
                out = {"text": str(data.get("result") or ""), "tool_calls": []}
        calls = []
        for c in out.get("tool_calls") or []:
            try:
                args = json.loads(c.get("args_json") or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = {"_unparseable_arguments": str(c.get("args_json"))[:500]}  # tool layer -> bad_arguments
            calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=str(c.get("name", "")), args=args))
        text = (out.get("text") or "").strip() or None
        if on_stream and text and not calls:
            on_stream("text", text)
        u = data.get("usage") or {}
        usage = Usage(input_tokens=(u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                      + (u.get("cache_creation_input_tokens") or 0),
                      output_tokens=u.get("output_tokens") or 0)
        used = list((data.get("modelUsage") or {}).keys())
        if used:
            self.model = used[0]
        return LLMResponse(text=text, tool_calls=calls, usage=usage, model=self.model,
                           cost_usd=data.get("total_cost_usd"))
