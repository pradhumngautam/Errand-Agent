"""Record one end-to-end booking whose book() call times out after actually succeeding.

    uv run python scripts/record_trace.py            # live model (same one the web app uses)
    uv run python scripts/record_trace.py --scripted # offline, scripted model

Writes every run's trace events, in order, to docs/trace-timeout-after-success.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from errand_agent.agent import AgentConfig
from errand_agent.clock import FakeClock
from errand_agent.llm.scripted import ScriptedLLM, call, say
from errand_agent.mocks.scenario import Scenario
from errand_agent.runtime import build_agent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "trace-timeout-after-success.jsonl"
TURNS = ["Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.",
         "The top-rated one, earliest slot please",
         "yes"]


def scripted_model(agent_ref: list) -> ScriptedLLM:
    def hold_first(view):
        c = agent_ref[0].workflows.active("u1").candidates[0]
        return call("hold_slot", provider_id=c["provider_id"], slot=c["matching_slots"][0])
    return ScriptedLLM([
        call("update_request", intent="book", service="plumber", date_start="2026-11-10", date_end="2026-11-15",
             latest_time="12:00", budget_max=1500),
        call("search_providers", service="plumber", area="Koramangala",
             date_range={"start": "2026-11-10", "end": "2026-11-15"}),
        say("Here are the best options."),
        hold_first, call("request_approval", action="book"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scripted", action="store_true", help="use the offline scripted model")
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    work = Path(tempfile.mkdtemp(prefix="errand-trace-"))
    agent_ref: list = []
    if args.scripted:
        llm = scripted_model(agent_ref)
    else:
        from errand_agent.llm.factory import make_live_llm
        llm = make_live_llm(cache_dir=str(work))
    agent = build_agent(work / "data", llm, scenario=Scenario(book_outcomes=["timeout_booked"]), clock=FakeClock(),
                        config=AgentConfig(trace_dir=str(work / "traces")))
    agent_ref.append(agent)
    run_ids = []
    for text in TURNS:
        reply = agent.handle_message(text)
        run_ids.append(reply.run_id)
        print(f"\nUSER  {text}\nAGENT {reply.text}\n      phase={reply.phase} trace={reply.run_id}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as out:
        for rid in run_ids:
            out.write((work / "traces" / f"{rid}.jsonl").read_text())
    events = [json.loads(line) for line in OUT.open()]
    books = [e for e in events if e["type"] == "tool_call" and e["tool"] == "book"]
    print(f"\nwrote {OUT.relative_to(ROOT)}: {len(events)} events, model {getattr(llm, 'model', '?')}, "
          f"{len(books)} book() call(s)")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
