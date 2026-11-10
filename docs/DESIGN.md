# Errand Agent: design note

**Shape.** One model drives the conversation through *intent tools* (`update_request`, `search_providers`, `hold_slot`, `request_approval`, memory tools). It has **no tool that books, cancels or reschedules**. Those three run only in application code, after the user confirms. A second, tool-less agent (the review checker) reads raw provider reviews and returns a digest, so free text from providers never enters the main model's context. Trade-off: the model can't take shortcuts, so a few extra turns buy approval, idempotency and verification that hold whatever the model does.

**State model.** Each errand is an explicit, persisted workflow: `IDLE → COLLECTING → PRESENTING → HOLDING → AWAITING_APPROVAL → EXECUTING → DONE | FAILED`. Illegal transitions are refused and traced. `EXECUTING` is reachable only from `AWAITING_APPROVAL`, and it is saved to SQLite before any side effect runs. The request is a typed record: `update_request` merges only the changed fields, so a correction keeps everything else. `missing()` counts stored preferences as known, so known facts are never asked again.

**Approval enforcement.** `request_approval` builds the card from authoritative state (the current hold and its quoted price), never from model arguments. The approval stores a SHA-256 fingerprint of action, provider, slot and price. It executes only when (a) the user presses the card's button or replies with a bare affirmative ("yes", "go ahead"; not "yes, and also cancel X"), and (b) the fingerprint still matches at execution time. Any other reply withdraws the card. Any change to provider, slot or price voids the approval and produces a new card. Guards also refuse unknown tools, cancel/reschedule intents the user never stated, and approvals whose action differs from the request. They rewrite unverified success claims such as "I've booked…".

**Idempotency and recovery.** The idempotency key is derived, not random (`book:{approval_id}:{hold_id}`), and a write-ahead `action_log` row commits with the `EXECUTING` mark *before* the provider is called. On a timeout or 5xx the executor never re-sends blindly. It asks `get_booking_status`, and re-sends with the same key only after the provider confirms nothing was booked. If the status service is unreachable, the outcome is `unknown`: the user is told so, nothing is re-sent, and the watcher settles it later. On startup, `recover()` reconciles every `pending/unknown` log row against the provider. A crash before the call resumes once; a crash after the call is recognised as done. Success is reported only from a verified booking record.

**Memory.** Every fact carries `source` (explicit or inferred), `confirmed` and `status` (active, superseded or forgotten). A value counts as explicit only if it appears in the user's own words, so an injected or hallucinated value is downgraded to an unconfirmed inference. Inferences are never used until the user confirms them, and they never override an explicit fact. Users can list, correct (which supersedes and keeps history) and forget facts. Context is capped at a fixed budget (10k tokens). The builder sends only active, usable, in-scope memories and trims the long history. It logs a **manifest per model call**: tokens per section, memory IDs included, and every excluded ID with its reason. Evals audit each prompt live against the memory table.

**Bounded execution.** Each run has hard limits: 8 model turns, 20 tool calls (plus a small reserve kept for status checks, so a limit can never hide whether we booked) and 90 s of wall-clock time. Every exit path (limit hit, model outage, tool error, unexpected exception) returns a plain reply saying what did and did not happen.

**Proactive follow-up.** A watcher polls `list_bookings`. Any transition into `cancelled_by_provider` triggers a notification, a replacement search within stored preferences, a hold and a *fresh* approval card. Polls that find nothing stay out of the trace.

**Observability.** Each run writes JSONL events: state transitions, context manifests, model calls (tokens, latency, estimated cost), tool calls (sanitised arguments and results, latency), approvals, guards and retries. The web UI streams them beside the chat.

**Trace excerpt** (`docs/trace-timeout-after-success.jsonl`, live model, run R-0003, after the user says "yes"):

```text
approval  AP-0001 approved via chat           # bare "yes" matched to the pending card
state     AWAITING_APPROVAL -> EXECUTING     # persisted before the side effect
state     action book EXECUTING key=book:AP-0001:H-0001   # write-ahead log row committed
tool_call book(H-0001, key) ok=false error=timeout       # provider timed out...
ambiguous_outcome book timeout                           # ...so success is NOT assumed
tool_call get_booking_status(key) -> BK-0004 confirmed   # source of truth: it did book
action_log key status=succeeded; approval completed      # no second book() call
state     EXECUTING -> DONE "booking verified"
run_end   "The provider's system timed out... It did go through, and I didn't send it twice."
```

Model `input_tokens` in this trace are what the local model runtime reports, including its own fixed prompt and cache reads. The budget applies to what the agent sends (`estimated_tokens` in each `context` event).
