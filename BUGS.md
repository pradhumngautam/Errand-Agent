# Real bugs hit while building

## 1. Approval invalidation silently rolled back

- **What broke:** When the material details changed after approval (for example the price was re-quoted), the executor refused to book, which was correct. But the approval stayed `approved` in the database instead of becoming `invalidated`.
- **Why:** `begin_execution` checked the fingerprint, wrote `invalidated`, then raised `ApprovalError`. It ran inside the same `BEGIN IMMEDIATE` transaction that also writes the action-log row, so the raise rolled the invalidation back too.
- **Fix:** Split it in two. `ApprovalManager.verify_material` runs outside the transaction and commits the invalidation on its own. `mark_executing` and the write-ahead log row then commit together atomically (`executor.py::_start`).
- **Regression tests:** `tests/test_executor.py::test_material_change_after_approval_invalidates` (3 cases) and `::test_price_drift_on_rehold_requires_fresh_approval`. All 4 fail when the old `_start` is restored and pass with the fix.

## 2. Provider cancellation of a new booking went unnoticed

- **What broke:** After the agent booked a plumber and the provider then cancelled it, `poll_events()` reported nothing. The proactive follow-up never fired, which is exactly the case it exists for.
- **Why:** The watcher only flagged bookings whose last snapshot status was `confirmed`. The snapshot was taken at startup and refreshed only on each poll, so a booking made since the last poll was unknown to it. When that booking became `cancelled_by_provider`, `known.get(id) == "confirmed"` was false.
- **Fix:** Detect any transition *into* `cancelled_by_provider` (`known.get(id) != "cancelled_by_provider"`), so bookings the snapshot has never seen count. Also record a booking in the watch table as soon as it is verified (`agent.py::poll_events`, `_after_book`).
- **Regression test:** `tests/test_agent.py::test_proactive_provider_cancellation` fails on the old code and passes now. It also checks that the user is not notified twice.

## 3. "Never before 10am" was not saved as an explicit preference

- **What broke:** The user said "From now on never before 10am please", and the model correctly called `remember_preference(earliest_time="10:00")`. The stored preference stayed at 09:00.
- **Why:** To stop prompt injection planting preferences, a value counts as explicit only if it appears in the user's own words. That check compared digit strings: `"10:00"` produced `["10", "00"]`, and `"00"` never appears in "10am". So the value was downgraded to an inference, and an inference never overrides an explicit fact, so nothing changed. The tool then told the model it had been "saved as an inference", which was also wrong.
- **Fix:** Compare numbers as numbers and ignore zero components. Strip thousands separators ("₹1,500" matches 1500) and treat "6pm" as 18. When an unstated value conflicts with an explicit fact, the tool now reports `not_stored` accurately (`tools.py::_stated_by_user`, `t_remember_preference`).
- **Regression test:** `tests/test_agent.py::test_memory_explicit_vs_downgraded_and_forget` fails on the old code and passes now. It covers explicit save, an unstated value that can't override an explicit fact, an unstated value stored as an unusable inference, and forgetting.

## 4. (Bonus) The live model named the wrong weekday for a slot

- **What broke:** Recording the end-to-end trace with the live model, the agent offered "Nisha Pipe Care, Tue 11 Nov 08:00", but 11 Nov 2026 is a Wednesday. The approval card, which is built in code, correctly said "Wed 11 Nov", so the chat and the card disagreed about the day.
- **Why:** Tool results carried slots only as ISO timestamps (`2026-11-11T08:00+05:30`). The model worked out the weekday itself and got it wrong. The system prompt gave only today's date.
- **Fix:** Weekday labels are now computed in code (`clock.slot_label`) and sent next to every slot: `slot_labels` in search results, `slot_label` in `hold_slot` and `list_bookings` results. The system prompt also carries a 7-day calendar and tells the model to use those labels and never compute a weekday.
- **Regression test:** `tests/test_agent.py::test_slots_reach_the_model_with_code_computed_weekdays`.
