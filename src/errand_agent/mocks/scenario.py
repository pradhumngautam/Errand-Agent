"""Configurable, seeded behaviour of the mock provider environment.

Every random choice the mocks make comes from a seeded RNG, and every
interesting failure can also be forced, so eval scenarios are reproducible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BookOutcome = Literal["success", "slot_taken", "timeout_booked", "timeout_not_booked"]


class Scenario(BaseModel):
    seed: int = 42

    # book(): forced outcomes are consumed in order; after that, the seeded RNG
    # draws with the contract's distribution (70% / 15% / 15% split in half).
    book_outcomes: list[BookOutcome] = Field(default_factory=list)
    book_probs: dict[BookOutcome, float] = Field(
        default_factory=lambda: {
            "success": 0.70,
            "slot_taken": 0.15,
            "timeout_booked": 0.075,
            "timeout_not_booked": 0.075,
        }
    )

    # hold_slot(): quoted price = search price + delta (configurable drift).
    hold_price_delta: int = 0
    hold_price_delta_providers: list[str] | None = None  # None = every provider
    hold_ttl_minutes: int = 10

    # cancel(): the first call per booking fails with a transient 503.
    cancel_fail_first: bool = True

    # reschedule(): these slots (ISO strings) always return slot_conflict.
    reschedule_conflict_slots: list[str] = Field(default_factory=list)

    # Generic failure injection: tool name -> queue of error codes raised on the
    # next calls (e.g. {"search_providers": ["unavailable_503"]}).
    # "timeout" raises ToolTimeout, codes ending in "_503" are transient.
    tool_failures: dict[str, list[str]] = Field(default_factory=dict)

    def wants_price_delta(self, provider_id: str) -> bool:
        if not self.hold_price_delta:
            return False
        return self.hold_price_delta_providers is None or provider_id in self.hold_price_delta_providers
