"""The mock environment honours the fixed tool contract."""

from __future__ import annotations

from collections import Counter

import pytest

from errand_agent.clock import FakeClock
from errand_agent.mocks.catalog import INJECTION_REVIEWS
from errand_agent.mocks.errors import ToolError, ToolTimeout, TransientToolError
from errand_agent.mocks.scenario import Scenario
from errand_agent.mocks.service import MockProviderService

WEEK = {"start": "2026-11-10", "end": "2026-11-16"}


def svc(tmp_path, scenario=None, name="p.db"):
    return MockProviderService(tmp_path / name, FakeClock(), scenario or Scenario())


def first_hold(s: MockProviderService, service="plumber", area="Koramangala"):
    for p in s.search_providers(service, area, WEEK)["providers"]:
        if p["available_slots"]:
            return p, s.hold_slot(p["provider_id"], p["available_slots"][0])
    raise AssertionError("no slots")


def test_search_returns_3_to_8_providers_with_required_fields(tmp_path):
    s = svc(tmp_path)
    for service in ["plumber", "electrician", "ac_servicing", "carpenter", "cleaning"]:
        res = s.search_providers(service, "Koramangala", WEEK)
        assert 3 <= len(res["providers"]) <= 8
        for p in res["providers"]:
            assert {"provider_id", "price", "rating", "available_slots", "attributes"} <= p.keys()


def test_world_is_deterministic_for_a_seed(tmp_path):
    a = svc(tmp_path, name="a.db").search_providers("plumber", "Indiranagar", WEEK)
    b = svc(tmp_path, name="b.db").search_providers("plumber", "Indiranagar", WEEK)
    assert a == b


def test_some_review_contains_prompt_injection(tmp_path):
    s = svc(tmp_path)
    ids = [p["provider_id"] for p in s.search_providers("plumber", "Koramangala", WEEK)["providers"]]
    texts = [r["text"] for pid in ids for r in s.get_provider(pid)["reviews"]]
    assert any(t in INJECTION_REVIEWS for t in texts)


def test_hold_lasts_10_minutes_and_price_drift_is_configurable(tmp_path):
    s = svc(tmp_path, Scenario(hold_price_delta=200))
    provider, hold = first_hold(s)
    assert hold["quoted_price"] == provider["price"] + 200
    s.clock.advance(minutes=10, seconds=1)
    with pytest.raises(ToolError) as e:
        s.book(hold["hold_id"], "k1")
    assert e.value.code == "hold_expired"


@pytest.mark.parametrize("outcome,booked", [("timeout_booked", True), ("timeout_not_booked", False)])
def test_timeout_after_possible_success(tmp_path, outcome, booked):
    s = svc(tmp_path, Scenario(book_outcomes=[outcome]))
    _, hold = first_hold(s)
    with pytest.raises(ToolTimeout):
        s.book(hold["hold_id"], "k1")
    status = s.get_booking_status(idempotency_key="k1")
    assert (status["status"] == "confirmed") is booked


def test_book_is_idempotent_per_key(tmp_path):
    s = svc(tmp_path, Scenario(book_outcomes=["success"]))
    _, hold = first_hold(s)
    first = s.book(hold["hold_id"], "k1")
    second = s.book(hold["hold_id"], "k1")
    assert first["booking_id"] == second["booking_id"] and second["idempotent_replay"]


def test_book_outcome_distribution_matches_contract(tmp_path):
    s = svc(tmp_path, Scenario(seed=7))
    counts = Counter(s._draw_book_outcome() for _ in range(4000))
    assert abs(counts["success"] / 4000 - 0.70) < 0.03
    assert abs(counts["slot_taken"] / 4000 - 0.15) < 0.03
    assert abs((counts["timeout_booked"] + counts["timeout_not_booked"]) / 4000 - 0.15) < 0.03


def test_cancel_first_call_503_then_retry_succeeds(tmp_path):
    s = svc(tmp_path)
    upcoming = next(b for b in s.list_bookings("u1")["bookings"] if b["status"] == "confirmed")
    with pytest.raises(TransientToolError):
        s.cancel(upcoming["booking_id"], "c1")
    assert s.cancel(upcoming["booking_id"], "c1")["status"] == "cancelled"
    assert s.cancel(upcoming["booking_id"], "c1")["idempotent_replay"]


def test_reschedule_conflict_on_configured_slot(tmp_path):
    s = svc(tmp_path)
    ac = next(b for b in s.list_bookings("u1")["bookings"] if b["service"] == "ac_servicing")
    saturday = [x for x in s.search_providers("ac_servicing", "Koramangala", {"start": "2026-11-14", "end": "2026-11-14"})
                ["providers"] if x["provider_id"] == ac["provider_id"]][0]["available_slots"]
    s.scenario.reschedule_conflict_slots = [saturday[0]]
    with pytest.raises(ToolError) as e:
        s.reschedule(ac["booking_id"], saturday[0], "r1")
    assert e.value.code == "slot_conflict"
    if len(saturday) > 1:
        assert s.reschedule(ac["booking_id"], saturday[1], "r2")["slot"] == saturday[1]


def test_list_bookings_has_seeded_history(tmp_path):
    bookings = svc(tmp_path).list_bookings("u1")["bookings"]
    assert {b["service"] for b in bookings} >= {"electrician", "ac_servicing", "plumber"}
    assert any(b["status"] == "completed" for b in bookings) and any(b["status"] == "confirmed" for b in bookings)


def test_provider_state_survives_restart(tmp_path):
    s = svc(tmp_path, Scenario(book_outcomes=["success"]))
    _, hold = first_hold(s)
    booking = s.book(hold["hold_id"], "k1")
    s.close()
    s2 = MockProviderService(tmp_path / "p.db", FakeClock(), Scenario())
    assert s2.get_booking_status(idempotency_key="k1")["booking_id"] == booking["booking_id"]


def test_failure_injection(tmp_path):
    s = svc(tmp_path, Scenario(tool_failures={"search_providers": ["unavailable_503", "internal_error"]}))
    with pytest.raises(TransientToolError):
        s.search_providers("plumber", "Koramangala", WEEK)
    with pytest.raises(ToolError) as e:
        s.search_providers("plumber", "Koramangala", WEEK)
    assert e.value.code == "internal_error"
    assert s.search_providers("plumber", "Koramangala", WEEK)["providers"]
