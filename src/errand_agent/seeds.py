"""Seed data: 55 stored memories (all statuses and sources) and a long prior conversation.

The mix is deliberate. Most memories are irrelevant to any single request,
some are forgotten or superseded, and some are inferred but unconfirmed.
Context-budget tests assert that only relevant, current, usable facts reach
the model.
"""

from __future__ import annotations

from datetime import timedelta

from errand_agent.clock import Clock, fmt_dt
from errand_agent.memory import MemoryStore
from errand_agent.store import AgentStore

# (key, value, scope, source, confirmed)
_ACTIVE = [
    ("area", "Koramangala", "global", "explicit", True),
    ("earliest_time", "09:00", "global", "explicit", True),
    ("preferred_provider", "ELE-KOR-01", "electrician", "explicit", True),
    ("budget_max", "1500", "plumber", "explicit", True),
    ("budget_max", "2500", "ac_servicing", "explicit", True),
    ("note_ac_brand", "Daikin split AC, 1.5 ton, two units", "ac_servicing", "explicit", True),
    ("note_access", "Flat 4B, ask security to call before coming up", "global", "explicit", True),
    ("note_pets", "Has a friendly dog; technician should be OK with pets", "global", "explicit", True),
    ("note_language", "Prefers English or Hindi", "global", "explicit", True),
    ("note_cleaning_products", "Eco-friendly cleaning products only", "cleaning", "explicit", True),
    ("note_carpentry_wood", "Prefers teak for repairs", "carpenter", "explicit", True),
    ("note_payment", "Pays by UPI after the job", "global", "explicit", True),
    ("note_kitchen_tap", "Kitchen tap is a Jaquar single-lever mixer", "plumber", "explicit", True),
    ("note_geyser", "Bathroom geyser is 15L Bajaj, installed 2021", "electrician", "explicit", True),
    ("note_parking", "Visitor parking available in basement B1", "global", "explicit", True),
    ("note_weekend", "Avoid Sunday mornings (family time)", "global", "explicit", True),
    ("note_invoice", "Wants a GST invoice for AC servicing", "ac_servicing", "explicit", True),
    ("note_shoes", "Ask technicians to use shoe covers", "global", "explicit", True),
    ("note_deep_clean", "Deep clean kitchen twice a year", "cleaning", "explicit", True),
    ("note_wardrobe", "Bedroom wardrobe hinge is loose", "carpenter", "explicit", True),
    ("note_inverter", "Has a 1kVA inverter; electrician should check it", "electrician", "explicit", True),
    ("note_water_pressure", "Low water pressure on the 4th floor in mornings", "plumber", "explicit", True),
    ("note_ac_gas", "AC gas was topped up in March", "ac_servicing", "explicit", True),
    ("note_balcony", "Balcony door needs weather stripping", "carpenter", "explicit", True),
    ("note_sofa", "Sofa shampoo with cleaning visits when possible", "cleaning", "explicit", True),
    ("note_contact", "Prefers chat over calls", "global", "explicit", True),
    ("note_warranty", "Prefers providers offering a warranty", "global", "explicit", True),
    ("note_switchboard", "Living room switchboard sparks occasionally", "electrician", "explicit", True),
    ("note_drain", "Bathroom drain clogs every few months", "plumber", "explicit", True),
    ("note_ac_remote", "Lost one AC remote, may need a replacement", "ac_servicing", "explicit", True),
    ("preferred_provider", "CLN-KOR-02", "cleaning", "inferred", True),        # inferred and confirmed
    ("latest_time", "18:00", "global", "inferred", True),                      # inferred and confirmed
    # inferred and NOT confirmed: must never reach the model
    ("preferred_provider", "PLU-KOR-03", "plumber", "inferred", False),
    ("budget_max", "2000", "cleaning", "inferred", False),
    ("note_morning_person", "Seems to prefer morning slots", "global", "inferred", False),
    ("note_brand_loyalty", "Might prefer premium providers", "global", "inferred", False),
    ("budget_max", "1200", "electrician", "inferred", False),
    ("note_weekday_only", "Possibly weekday-only bookings", "global", "inferred", False),
]
# (key, value, scope): written and then forgotten
_FORGOTTEN = [
    ("note_phone", "Alternate phone 98450 12345", "global"),
    ("note_office", "Works from the Whitefield office on Tuesdays", "global"),
    ("note_old_flat", "Old flat in Indiranagar 12th Main", "global"),
    ("preferred_provider", "PLU-KOR-05", "plumber"),
    ("note_secret_gate_code", "Gate code 4412", "global"),
    ("note_ex_landlord", "Landlord Mr. Rao handles plumbing above ₹5000", "plumber"),
]
# (key, old, new, scope): old value superseded by a correction
_SUPERSEDED = [
    ("area", "HSR Layout", None, "global"),          # moved to Koramangala (active above)
    ("earliest_time", "08:00", None, "global"),      # corrected to 09:00 (active above)
    ("budget_max", "1000", None, "plumber"),         # corrected to 1500 (active above)
    ("note_ac_units", "One AC unit", "Two AC units installed", "ac_servicing"),
    ("note_language_old", "Kannada only", "English, Hindi or Kannada", "global"),
    ("note_cleaning_day", "Saturdays", "Fridays", "cleaning"),
]

FORGOTTEN_VALUES = [v for _, v, _ in _FORGOTTEN]


def seed_memories(memory: MemoryStore, clock: Clock) -> None:
    base = clock.now() - timedelta(days=200)
    t = 0

    def ts() -> str:
        nonlocal t
        t += 1
        return fmt_dt(base + timedelta(days=t * 3))

    # superseded entries first, so the corrected active values supersede them
    for key, old, new, scope in _SUPERSEDED:
        memory.remember(key, old, scope=scope, created_at=ts())
        if new:
            memory.remember(key, new, scope=scope, created_at=ts())
    for key, value, scope in _FORGOTTEN:
        m = memory.remember(key, value, scope=scope, created_at=ts())
        memory.forget(memory_id=m.memory_id)
    for key, value, scope, source, confirmed in _ACTIVE:
        memory.remember(key, value, scope=scope, source=source, confirmed=confirmed,  # type: ignore[arg-type]
                        created_at=ts(), evidence="seeded history")


_PAST_TOPICS = [
    ("Can you get someone to fix the bathroom drain?", "Sure, I found 3 plumbers in Koramangala this week."),
    ("The second one please, Thursday 10am.", "Booked after your confirmation. Booking verified."),
    ("What's my AC servicing date?", "It's on Friday at 15:00 with your usual AC provider."),
    ("I need a deep clean of the kitchen next month.", "Noted. Which week works for you?"),
    ("Remember I live in Koramangala now, not HSR.", "Updated: your area is now Koramangala."),
    ("Electrician for the switchboard, not before 9.", "Found 4 electricians with slots after 09:00."),
    ("Too expensive, anything under 800?", "Two options are under ₹800. Here they are."),
    ("Thanks, that's all for today.", "Happy to help. Have a good evening!"),
]


def seed_conversation(store: AgentStore, clock: Clock, user_id: str = "u1", exchanges: int = 90) -> None:
    start = clock.now() - timedelta(days=180)
    rows = []
    for i in range(exchanges):
        q, a = _PAST_TOPICS[i % len(_PAST_TOPICS)]
        when = fmt_dt(start + timedelta(days=i * 2))
        rows.append((user_id, "user", f"{q} (session {i // 4 + 1})", "seed", when))
        rows.append((user_id, "assistant", a, "seed", when))
    with store.tx() as db:
        db.executemany("INSERT INTO conversation(user_id, role, text, run_id, ts) VALUES (?,?,?,?,?)", rows)
