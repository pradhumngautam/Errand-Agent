"""Runs the eval suite (and regenerates evals/RESULTS.md), then checks the invariants can fail."""

from __future__ import annotations

from pathlib import Path

import pytest

from errand_agent.evals.harness import Harness
from errand_agent.evals.invariants import (
    bookings_match_approval, no_duplicate_bookings, success_only_when_verified,
    unusable_memory_never_sent, untrusted_text_never_reaches_model,
)
from errand_agent.evals.run import run_all, write
from errand_agent.evals.scenarios import SCENARIOS, pick_first, search_plumbers
from errand_agent.llm.scripted import say
from errand_agent.mocks.catalog import INJECTION_REVIEWS

ROOT = Path(__file__).resolve().parents[1]


def test_eval_suite_passes_and_writes_table():
    rows = run_all()
    write(rows, ROOT / "evals")
    assert len(rows) >= 20
    failed = [f"{r.scenario}: {r.notes}" for r in rows if not r.passed]
    assert not failed, "\n".join(failed)


def test_required_categories_are_covered():
    names = {s.name for s in SCENARIOS}
    for required in ["adversarial_user_fake_approval", "prompt_injection_in_reviews", "price_changed_after_approval",
                     "slot_changed_after_approval", "timeout_after_success", "reschedule_slot_conflict",
                     "repeated_confirmation", "memory_correction_and_inspection", "memory_forget",
                     "restart_after_book_before_record", "proactive_provider_cancellation",
                     "context_budget_with_long_history", "limit_model_turns", "search_transient_failure"]:
        assert required in names


# ------------------------------------------------------- the invariants can fail
# An invariant that never fails proves nothing. Each test breaks one rule behind the
# agent's back and checks that the matching invariant notices.

@pytest.fixture
def h(tmp_path) -> Harness:
    return Harness(tmp_path)


def test_detects_booking_made_without_approval(h):
    search_plumbers(h)
    opt = h.first_option()
    hold = h.service.hold_slot(**opt)
    h.service.book(hold_id=hold["hold_id"], idempotency_key="rogue-key")   # bypasses approvals entirely
    assert not bookings_match_approval(h).ok


def test_detects_duplicate_booking(h):
    search_plumbers(h)
    pick_first(h)
    h.user("yes")
    b = h.agent_bookings()[0]
    h.service.db.execute("INSERT INTO bookings SELECT 'BK-DUP', user_id, provider_id, provider_name, service, slot, "
                         "price, status, hold_id, NULL, created_at, updated_at FROM bookings WHERE booking_id = ?",
                         (b["booking_id"],))
    h.service.db.execute("UPDATE bookings SET idempotency_key = ? WHERE booking_id = 'BK-DUP'",
                         (b["idempotency_key"] + "-copy",))
    h.agent.store.db.execute("INSERT INTO action_log SELECT idempotency_key || '-copy', approval_id, action, request, "
                             "status, result, created_at, updated_at FROM action_log WHERE idempotency_key = ?",
                             (b["idempotency_key"],))
    assert not no_duplicate_bookings(h).ok


def test_detects_unverified_success_claim(h):
    h.script(say("Your plumber is booked."))
    reply = h.agent.handle_message("book a plumber")
    reply.text = "I've booked your plumber."     # as if the claim guard were missing
    h._record("user", "book a plumber", reply)
    assert not success_only_when_verified(h).ok


def test_detects_raw_review_text_sent_to_model(h):
    h.llm.calls.append({"system": INJECTION_REVIEWS[0], "messages": [], "tools": []})
    assert not untrusted_text_never_reaches_model(h).ok


def test_detects_forgotten_fact_in_prompt(h):
    forgotten = next(m for m in h.agent.memory.all() if m.status == "forgotten")
    h.script(say("ok"))
    h.llm.complete(f"- {forgotten.memory_id}: {forgotten.key} = {forgotten.value}", [], [])
    assert not unusable_memory_never_sent(h).ok
