"""Terminal chat for trying the live model:  uv run errand-chat [--fresh] [--replay CASSETTE] [--record CASSETTE]

Commands: /approve <id>   /poll (proactive check)   /cancel-by-provider <booking_id>
          /memories       /quit
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

from errand_agent.agent import AgentConfig
from errand_agent.llm.base import LLMError


def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cli")
    ap.add_argument("--fresh", action="store_true", help="wipe local state and reseed")
    ap.add_argument("--record", help="record model responses to this cassette")
    ap.add_argument("--replay", help="replay a recorded cassette instead of calling the model")
    args = ap.parse_args()

    from errand_agent.llm.recording import RecordingLLM, ReplayLLM
    from errand_agent.runtime import build_agent

    if args.fresh and Path(args.data).exists():
        shutil.rmtree(args.data)
    try:
        if args.replay:
            llm = ReplayLLM(args.replay)
        else:
            from errand_agent.llm.factory import make_live_llm
            llm = make_live_llm(cache_dir="data")
            if args.record:
                llm = RecordingLLM(llm, args.record)
    except LLMError as exc:
        raise SystemExit(f"Model setup failed: {exc}")
    agent = build_agent(args.data, llm, config=AgentConfig(trace_dir=f"{args.data}/traces"))
    print(f"Errand Agent ({llm.model}). Type /quit to exit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text.startswith("/approve "):
            reply = agent.confirm(text.split(maxsplit=1)[1])
        elif text == "/poll":
            replies = agent.poll_events()
            for r in replies:
                print(f"agent (proactive)> {r.text}\n")
            print("" if replies else "(nothing new)\n")
            continue
        elif text.startswith("/cancel-by-provider "):
            agent.service.simulate_provider_cancellation(text.split(maxsplit=1)[1])
            print("(provider cancelled it; run /poll)\n")
            continue
        elif text == "/memories":
            for m in agent.memory.all():
                print(f"  {m.memory_id} {m.key}[{m.scope}]={m.value} {m.source} confirmed={m.confirmed} {m.status}")
            continue
        else:
            reply = agent.handle_message(text)
        print(f"agent> {reply.text}")
        if reply.approval_card:
            print(f"       [approval {reply.approval_card['approval_id']}]")
        print(f"       (phase {reply.phase} · trace {reply.trace_path})\n")


if __name__ == "__main__":
    main()
