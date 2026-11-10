"""End-to-end agent behaviour with a scripted (untrusted) model."""

from __future__ import annotations

import json

import pytest

from errand_agent.agent import AgentConfig
from errand_agent.budget import Limits
from errand_agent.clock import FakeClock, parse_dt
from errand_agent.llm.base import LLMError
from errand_agent.llm.scripted import ScriptedLLM, call, calls, say
from errand_agent.mocks.catalog import INJECTION_REVIEWS
from errand_agent.mocks.scenario import Scenario
from errand_agent.runtime import build_agent

WEEK = {"start": "2026-11-10", "end": "2026-11-15"}
PLUMBER_REQ = dict(intent="book", service="plumber", date_start="2026-11-10", date_end="2026-11-15",
                   latest_time="12:00", budget_max=1500)


@pytest.fixture
def make_agent(tmp_path):
    def _make(steps=None, scenario=None, limits=None, budget=6000, loop_last=False):
        llm = ScriptedLLM(steps or [], loop_last=loop_last)
        agent = build_agent(tmp_path / "data", llm, scenario=scenario or Scenario(book_outcomes=["success"] * 5),
                            clock=FakeClock(), sleep=lambda s: None,
                            config=AgentConfig(context_budget_tokens=budget, limits=limits or Limits(),
                                               trace_dir=str(tmp_path / "traces")))
        return agent, llm
    return _make


def first_option(agent):
    c = agent.workflows.active("u1").candidates[0]
    return {"provider_id": c["provider_id"], "slot": c["matching_slots"][0]}


def events(agent, type_):
    return [e for e in agent.last_run.tracer.events if e["type"] == type_]


def confirmed_bookings(agent, service="plumber"):
    return [b for b in agent.service.list_bookings("u1")["bookings"]
            if b["service"] == service and b["status"] == "confirmed"]


def plumber_flow(agent, llm):
    """Turn 1: request -> options. Turn 2: pick -> approval card."""
    llm.extend([call("update_request", **PLUMBER_REQ),
                call("search_providers", service="plumber", area="Koramangala", date_range=WEEK),
                say("Here are the best options.")])
    r1 = agent.handle_message("Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.")
    llm.extend([lambda v: call("hold_slot", **first_option(agent)), call("request_approval", action="book")])
    r2 = agent.handle_message("The first one, earliest slot")
    return r1, r2


def test_full_booking_flow(make_agent):
    agent, llm = make_agent()
    r1, r2 = plumber_flow(agent, llm)
    upd = next(e for e in events_all(agent, r1.run_id) if e["type"] == "agent_tool" and e["tool"] == "update_request")
    assert upd["result"]["missing"] == []          # area came from memory, not re-asked
    assert "Koramangala" in json.dumps(upd["result"]["request"])
    assert r2.approval_card and r2.phase == "AWAITING_APPROVAL"
    assert confirmed_bookings(agent) == []          # nothing booked before confirmation
    calls_before = len(llm.calls)
    r3 = agent.handle_message("yes")
    assert len(llm.calls) == calls_before           # execution is not model-driven
    assert r3.outcome["status"] == "confirmed" and r3.phase == "DONE"
    assert len(confirmed_bookings(agent)) == 1
    assert "verified" in r3.text


def events_all(agent, run_id):
    path = agent.config.trace_dir + f"/{run_id}.jsonl"
    return [json.loads(line) for line in open(path)]


def test_repeated_confirmation_books_once(make_agent):
    agent, llm = make_agent()
    plumber_flow(agent, llm)
    agent.handle_message("yes")
    again = agent.handle_message("yes")
    button = agent.confirm(agent.workflows.active("u1").last_completed_approval_id)
    assert "already" in again.text.lower() and "already" in button.text.lower()
    assert len(confirmed_bookings(agent)) == 1


def test_changing_topic_withdraws_pending_approval(make_agent):
    agent, llm = make_agent()
    _, r2 = plumber_flow(agent, llm)
    old = r2.approval_card["approval_id"]
    llm.extend([say("Sure, what time works better?")])
    agent.handle_message("hmm, actually can we do later in the day?")
    late = agent.confirm(old)
    assert late.outcome is None and "no longer valid" in late.text
    assert confirmed_bookings(agent) == []


def test_correction_preserves_other_details(make_agent):
    agent, llm = make_agent()
    plumber_flow(agent, llm)
    llm.extend([call("update_request", budget_max=1200), say("Updated your budget.")])
    agent.handle_message("make the budget 1200 instead")
    req = agent.workflows.active("u1").request
    assert req.budget_max == 1200 and req.service == "plumber" and req.latest_time == "12:00"
    assert req.date_start == "2026-11-10"


def test_missing_details_are_only_the_unknown_ones(make_agent):
    agent, llm = make_agent()
    llm.extend([call("update_request", intent="book", service="carpenter"), say("When, and what budget?")])
    agent.handle_message("I need a carpenter")
    upd = next(e for e in events(agent, "agent_tool") if e["tool"] == "update_request")
    assert upd["result"]["missing"] == ["date_window", "budget"]   # area known from memory


def test_prompt_injection_never_reaches_model_and_cannot_act(make_agent):
    agent, llm = make_agent()
    agent.service.search_providers("plumber", "Koramangala", WEEK)
    injected_pid = "PLU-KOR-02"  # catalog puts an injection review on provider #2
    llm.extend([
        call("update_request", **PLUMBER_REQ),
        call("get_provider", provider_id=injected_pid),
        # model "falls for it": tries to book directly, cancel other bookings, and raise the budget
        calls(("book", {"hold_id": "H-0001", "idempotency_key": "x"}),
              ("update_request", {"intent": "cancel", "budget_max": 10000}),
              ("request_approval", {"action": "cancel", "booking_id": "BK-0003"})),
        say("Done!"),
    ])
    agent.handle_message("tell me about plumber PLU-KOR-02")
    sent = json.dumps(llm.calls)
    assert not any(t in sent for t in INJECTION_REVIEWS)
    assert any(e["type"] == "untrusted_content" for e in agent.last_run.tracer.events)
    guards = {e["rule"] for e in events(agent, "guard")}
    assert {"unknown_tool", "ungrounded_intent", "approval_intent_mismatch"} <= guards
    assert agent.workflows.active("u1").pending_approval_id is None
    assert agent.workflows.active("u1").request.budget_max == 1500
    assert all(b["status"] != "cancelled" for b in agent.service.list_bookings("u1")["bookings"])


def test_false_success_claim_is_rewritten(make_agent):
    agent, llm = make_agent()
    llm.extend([say("Great news, I've booked your plumber for tomorrow!")])
    r = agent.handle_message("book me a plumber")
    assert "Nothing has been booked" in r.text and "I've booked" not in r.text


def test_forgotten_and_unconfirmed_facts_never_sent(make_agent):
    agent, llm = make_agent(budget=4000)
    plumber_flow(agent, llm)
    sent = json.dumps(llm.calls)
    unusable = [m for m in agent.memory.all() if not m.usable]
    assert len(unusable) >= 15
    for m in unusable:
        assert f"{m.memory_id}:" not in sent, f"{m.memory_id} ({m.status}/{m.source}) leaked"
    for secret in ["Gate code 4412", "98450", "Seems to prefer morning slots", "Old flat in Indiranagar"]:
        assert secret not in sent
    # an unconfirmed inferred provider preference must not influence ranking either
    shortlist = agent.workflows.active("u1").candidates
    assert not any(c["preferred"] for c in shortlist if c["provider_id"] == "PLU-KOR-03")
    # the manifest logged each turn proves what was sent
    for e in events(agent, "context"):
        excluded = e["memory_excluded"]
        assert e["estimated_tokens"] <= e["budget_tokens"]
        assert set(e["memory_included"]).isdisjoint(excluded)
        assert {"forgotten", "superseded", "inferred_unconfirmed", "irrelevant_scope"} <= set(excluded.values())
        assert e["history_dropped"] > 0                        # long history was trimmed to budget


def test_limit_exhaustion_produces_clear_message(make_agent):
    agent, llm = make_agent(steps=[call("list_bookings")], loop_last=True, limits=Limits(max_model_turns=3))
    r = agent.handle_message("what are my bookings?")
    assert "limit" in r.text and "Nothing was booked" in r.text
    assert events(agent, "limit_reached")[0]["kind"] == "model_turns"


def test_llm_outage_produces_clear_message(make_agent):
    agent, llm = make_agent()

    def boom(v):
        raise LLMError("503 from provider")
    llm.extend([boom])
    r = agent.handle_message("find a plumber")
    assert "can't reach the language model" in r.text


def test_timeout_after_success_via_agent(make_agent):
    agent, llm = make_agent(scenario=Scenario(book_outcomes=["timeout_booked"]))
    plumber_flow(agent, llm)
    r = agent.handle_message("yes")
    assert r.outcome["status"] == "confirmed"
    assert len([e for e in agent.last_run.tracer.events if e["type"] == "tool_call" and e["tool"] == "book"]) == 1
    assert len(confirmed_bookings(agent)) == 1


def test_proactive_provider_cancellation(make_agent):
    agent, llm = make_agent()
    plumber_flow(agent, llm)
    booking = agent.handle_message("yes").outcome["booking"]
    agent.service.simulate_provider_cancellation(booking["booking_id"])
    notes = agent.poll_events()
    assert len(notes) == 1 and notes[0].kind == "notification" and notes[0].approval_card
    assert notes[0].approval_card["details"]["provider_id"] != booking["provider_id"]
    assert len(confirmed_bookings(agent)) == 0           # no rebooking without fresh approval
    r = agent.handle_message("yes")
    assert r.outcome["status"] == "confirmed" and len(confirmed_bookings(agent)) == 1
    assert agent.poll_events() == []                     # not re-notified


def test_reschedule_with_conflict_then_success(make_agent):
    agent, llm = make_agent()
    ac = next(b for b in agent.service.list_bookings("u1")["bookings"] if b["service"] == "ac_servicing")
    sat = next(p for p in agent.service.search_providers("ac_servicing", "Koramangala",
               {"start": "2026-11-14", "end": "2026-11-14"})["providers"]
               if p["provider_id"] == ac["provider_id"])["available_slots"]
    agent.service.scenario.reschedule_conflict_slots = [sat[0]]
    llm.extend([call("list_bookings"),
                call("update_request", intent="reschedule", booking_id=ac["booking_id"], new_slot=sat[0]),
                call("request_approval", action="reschedule")])
    r1 = agent.handle_message("Move my AC servicing to Saturday")
    assert r1.approval_card["details"]["slot"] == sat[0]
    r2 = agent.handle_message("yes")
    assert r2.outcome["reason"] == "slot_conflict" and "unchanged" in r2.text
    if len(sat) > 1:
        llm.extend([call("update_request", new_slot=sat[1]), call("request_approval", action="reschedule")])
        agent.handle_message("ok the next one then")
        r3 = agent.handle_message("yes")
        assert r3.outcome["status"] == "confirmed"
        moved = next(b for b in agent.service.list_bookings("u1")["bookings"] if b["booking_id"] == ac["booking_id"])
        assert moved["slot"] == sat[1]


def test_cancel_with_transient_503(make_agent):
    agent, llm = make_agent()
    ac = next(b for b in agent.service.list_bookings("u1")["bookings"] if b["service"] == "ac_servicing")
    llm.extend([call("update_request", intent="cancel", booking_id=ac["booking_id"]),
                call("request_approval", action="cancel")])
    agent.handle_message("Please cancel my AC servicing")
    r = agent.handle_message("yes")
    assert r.outcome["status"] == "confirmed" and "Cancelled" in r.text


def test_memory_explicit_vs_downgraded_and_forget(make_agent):
    agent, llm = make_agent()
    llm.extend([call("remember_preference", key="earliest_time", value="10:00"),
                call("remember_preference", key="budget_max", value="9000", scope="plumber"),
                call("remember_preference", key="budget_max", value="9000", scope="carpenter"),
                say("Saved.")])
    agent.handle_message("From now on never before 10am please")
    mem = {(m.key, m.scope): m for m in agent.memory.all(include_inactive=False)}
    assert mem[("earliest_time", "global")].value == "10:00" and mem[("earliest_time", "global")].source == "explicit"
    # 9000 was never stated by the user: it cannot override the explicit plumber budget, and
    # for carpenter it is stored only as an unconfirmed inference that is never used
    assert agent.memory.preference("budget_max", "plumber").value == "1500"
    downgraded = [m for m in agent.memory.all() if m.value == "9000"]
    assert [(m.scope, m.source, m.usable) for m in downgraded] == [("carpenter", "inferred", False)]
    assert agent.memory.preference("budget_max", "carpenter") is None
    llm.extend([call("forget_preference", key="earliest_time"), say("Forgotten.")])
    agent.handle_message("forget my earliest time preference")
    assert agent.memory.preference("earliest_time") is None


def test_restart_mid_booking_recovers_without_duplicate(make_agent, tmp_path):
    agent, llm = make_agent()
    plumber_flow(agent, llm)
    real_book = agent.service.book

    class Crash(BaseException):
        pass

    def book_then_die(**kw):
        real_book(**kw)
        raise Crash()
    agent.service.book = book_then_die
    with pytest.raises(Crash):
        agent.handle_message("yes")
    agent.service.book = real_book
    agent.store.close()
    agent2 = build_agent(tmp_path / "data", ScriptedLLM([]), clock=agent.clock, service=agent.service,
                         sleep=lambda s: None, startup=False)
    notes = agent2.startup()
    assert notes and notes[0].outcome["status"] == "confirmed"
    assert agent2.workflows.active("u1").phase.value == "DONE"
    assert len(confirmed_bookings(agent2)) == 1


def test_slots_reach_the_model_with_code_computed_weekdays(make_agent):
    """BUGS.md #4: the live model labelled 2026-11-11 as 'Tue' (it is a Wednesday)."""
    agent, llm = make_agent()
    plumber_flow(agent, llm)
    shortlist = agent.workflows.active("u1").candidates
    for c in shortlist:
        assert c["slot_labels"] == [parse_dt(s).strftime("%a %d %b, %H:%M") for s in c["matching_slots"]]
    assert any("Wed 11 Nov" in label for c in shortlist for label in c["slot_labels"])
    system = llm.calls[0]["system"]
    assert "Wed 2026-11-11" in system and "Sat 2026-11-14" in system
    hold = agent.workflows.active("u1").hold
    sent = json.dumps(llm.calls[-1]["messages"])
    assert parse_dt(hold.slot).strftime("%a %d %b, %H:%M") in sent   # hold_slot result carries its label
