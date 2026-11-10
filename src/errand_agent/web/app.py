"""Local demo server: chat on the left, live trace on the right, demo controls.

    uv run errand-web            ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Iterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from errand_agent.agent import Agent, AgentConfig, AgentReply
from errand_agent.clock import OffsetClock
from errand_agent.llm.base import LLMClient, LLMError
from errand_agent.mocks.scenario import Scenario
from errand_agent.runtime import build_agent

STATIC = Path(__file__).parent / "static"
DATA = Path(os.environ.get("ERRAND_DATA", "data/web"))
WATCH_SECONDS = float(os.environ.get("ERRAND_WATCH_SECONDS", "8"))


class Hub:
    """Owns the agent. One lock serialises every agent call (chat, button, watcher, demo controls)."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.events: list[dict] = []
        self.epoch = uuid.uuid4().hex[:8]   # changes on every reset; the page resyncs its trace when it does
        self.notifications: list[dict] = []
        self.llm: LLMClient | None = None
        self.llm_error: str | None = None
        self.agent: Agent | None = None
        self.clock = OffsetClock()

    def _listener(self, event: dict) -> None:
        self.events.append({**event, "gseq": len(self.events) + 1})

    def ensure_llm(self) -> None:
        if self.llm is None and self.llm_error is None:
            try:
                from errand_agent.llm.factory import make_live_llm
                DATA.mkdir(parents=True, exist_ok=True)
                self.llm = make_live_llm(cache_dir=str(DATA.parent))
            except LLMError as exc:
                self.llm_error = str(exc)

    def reset(self) -> None:
        with self.lock:
            if self.agent:
                self.agent.store.close()
                self.agent.service.close()
            shutil.rmtree(DATA, ignore_errors=True)
            self.events.clear()
            self.epoch = uuid.uuid4().hex[:8]
            self.notifications.clear()
            self.clock = OffsetClock()
            self.ensure_llm()
            if self.llm is None:
                self.agent = None
                return
            self.agent = build_agent(DATA, self.llm, scenario=Scenario(seed=int(os.environ.get("ERRAND_SEED", "42"))),
                                     clock=self.clock, trace_listener=self._listener,
                                     config=AgentConfig(trace_dir=str(DATA / "traces")))

    def require(self) -> Agent:
        if self.agent is None:
            raise HTTPException(503, self.llm_error or "agent not ready")
        return self.agent

    def watch_forever(self) -> None:
        while True:
            time.sleep(WATCH_SECONDS)
            try:
                with self.lock:
                    if self.agent is None:
                        continue
                    for r in self.agent.poll_events():
                        self.notifications.append(r.model_dump())
            except Exception as exc:  # noqa: BLE001 - the watcher must never die
                self.events.append({"type": "error", "source": "watcher", "error": str(exc),
                                    "gseq": len(self.events) + 1, "run_id": "-", "seq": 0, "elapsed_ms": 0})


hub = Hub()
app = FastAPI(title="Errand Agent")


@app.on_event("startup")
def _startup() -> None:
    load_dotenv(Path.cwd() / ".env")
    hub.reset()
    threading.Thread(target=hub.watch_forever, daemon=True).start()


class ChatIn(BaseModel):
    text: str


class ConfirmIn(BaseModel):
    approval_id: str


class DemoIn(BaseModel):
    action: str
    value: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/chat")
def chat(body: ChatIn) -> AgentReply:
    with hub.lock:
        return hub.require().handle_message(body.text)


@app.post("/api/confirm")
def confirm(body: ConfirmIn) -> AgentReply:
    with hub.lock:
        return hub.require().confirm(body.approval_id)


def _sse(work: Callable[[Callable[[dict], None]], AgentReply]) -> StreamingResponse:
    """Run `work(on_event)` in a worker thread and stream its events as Server-Sent Events.
    The final message is {"type": "reply", "reply": AgentReply}."""
    q: queue.Queue = queue.Queue()
    done = object()

    def runner() -> None:
        try:
            with hub.lock:
                reply = work(q.put)
            q.put({"type": "reply", "reply": reply.model_dump()})
        except HTTPException as exc:
            q.put({"type": "fatal", "error": exc.detail})
        except Exception as exc:  # noqa: BLE001
            q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            q.put(done)

    threading.Thread(target=runner, daemon=True).start()

    def gen() -> Iterator[str]:
        while True:
            item = q.get()
            if item is done:
                return
            yield f"data: {json.dumps(item, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn) -> StreamingResponse:
    return _sse(lambda emit: hub.require().handle_message(body.text, on_event=emit))


@app.post("/api/confirm/stream")
def confirm_stream(body: ConfirmIn) -> StreamingResponse:
    return _sse(lambda emit: hub.require().confirm(body.approval_id, on_event=emit))


@app.get("/api/events")
def events(after: int = 0) -> dict:
    return {"events": hub.events[after:], "total": len(hub.events), "epoch": hub.epoch}


@app.get("/api/notifications")
def notifications(after: int = 0) -> dict:
    return {"items": hub.notifications[after:], "total": len(hub.notifications)}


@app.get("/api/state")
def state() -> dict:
    with hub.lock:
        if hub.agent is None:
            return {"ready": False, "error": hub.llm_error}
        a = hub.agent
        wf = a.workflows.active(a.user_id)
        bookings = a.service.list_bookings(a.user_id)["bookings"]
        mems = a.memory.all()
        sc = a.service.scenario
        return {
            "ready": True, "model": a.llm.model, "now": a.clock.now().strftime("%a %d %b %H:%M"),
            "clock_offset_min": int(hub.clock.offset.total_seconds() // 60),
            "workflow": {"id": wf.workflow_id, "phase": wf.phase.value, "origin": wf.origin,
                         "request": wf.request.model_dump(exclude_none=True),
                         "hold": wf.hold.model_dump() if wf.hold else None,
                         "pending_approval_id": wf.pending_approval_id},
            "bookings": bookings,
            "memories": [m.model_dump() for m in mems],
            "scenario": {"next_book_outcomes": list(sc.book_outcomes), "hold_price_delta": sc.hold_price_delta},
            "context_calibration": round(a.context.calibration, 3),
        }


@app.post("/api/demo")
def demo(body: DemoIn) -> dict:
    with hub.lock:
        if body.action == "reset":
            hub.reset()
            return {"ok": True, "note": "fresh state"}
        a = hub.require()
        sc = a.service.scenario
        if body.action == "next_book":
            sc.book_outcomes = [body.value or "success"]
            return {"ok": True, "note": f"next book() -> {body.value}"}
        if body.action == "price_drift":
            sc.hold_price_delta = int(body.value or 0)
            return {"ok": True, "note": f"holds now quote search price +₹{sc.hold_price_delta}"}
        if body.action == "advance_clock":
            hub.clock.advance(minutes=float(body.value or 11))
            return {"ok": True, "note": f"clock +{body.value or 11} min"}
        if body.action == "provider_cancel":
            confirmed = [b for b in a.service.list_bookings(a.user_id)["bookings"] if b["status"] == "confirmed"]
            target = body.value or (max(confirmed, key=lambda b: b["booking_id"])["booking_id"] if confirmed else None)
            if not target:
                return {"ok": False, "note": "no confirmed booking to cancel"}
            a.service.simulate_provider_cancellation(target)
            return {"ok": True, "note": f"provider cancelled {target}; the watcher will notice within "
                                        f"{WATCH_SECONDS:.0f}s"}
        if body.action == "poll_now":
            items = [r.model_dump() for r in a.poll_events()]
            hub.notifications.extend(items)
            return {"ok": True, "note": f"{len(items)} notification(s)"}
        raise HTTPException(400, f"unknown action {body.action}")


def main() -> None:
    import uvicorn
    uvicorn.run("errand_agent.web.app:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
                log_level="warning")
