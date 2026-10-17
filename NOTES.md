# FreshKart take-home: notes and answers

Written answers to the "Think about it" questions, design choices and measurements for each exercise.
Code lives in `bot.py` (look for `# Ex N:` comments), and the tests are in `tests/test_app.py` (look for `# ---- Ex N` sections).

How to check everything:

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q          # offline tests use a fake LLM; live tests run when .env has OPENROUTER_API_KEY
grep -nE "pydantic|BaseModel|^class |with_structured_output" bot.py app.py   # must print nothing
```

## Phase 0: test harness

The fake LLM in `tests/test_app.py` stands in for `ChatOpenAI`. It does these things:

- It answers each kind of prompt (intent, language, card, checker, lane) with a canned answer. A test can override any answer.
- It returns an `AIMessage` with `response_metadata["model_name"]` and `usage_metadata`, the way a real model does.
- It can be told to fail for one model, or for all models.
- It records every call, so the tests can check what each prompt contained and how many LLM calls were made.
- It is a `RunnableSequence`, so `configurable_alternatives` works on it.

Because of this fake, almost every acceptance test runs without an API key.

## Ex1: PII shield

- **Concept:** a `RunnableLambda` (`mask_pii`) inside `RunnablePassthrough.assign(masked_input=..., pii_count=..., chat_history=...)`, placed as the first station of `router`. The intent prompt and the lane prompts read `masked_input`. The intent prompt gets it through a small adapter, `masked = RunnableLambda(lambda x: {"input": x["masked_input"]})`.
- **Routing does not change.** `order_id_from`, `has_safety` and `claimed_amount` still read the original `input`. The order ids (`FK1001`) and amounts (`₹520`) are too short for the 10/12–16 digit patterns.
- **Regex order:** email runs before UPI, because both contain `@`. Phone runs before card, because a `+91 98450 12345` phone has 12 digits and the card pattern would otherwise take it. The phone pattern has digit lookarounds, so it never matches inside a card number.
- **Beyond the spec:** earlier turns in `chat_history` are masked too. Otherwise a phone number from turn 1 would still reach the lane prompt in turn 2. Order ids survive masking, so memory still finds them. The stored history keeps the raw text in memory, but only the masked copy ever goes into a prompt.
- **Proof** (`test_x1_intent_prompt_is_masked_but_routing_unchanged`, `test_no_llm_call_ever_sees_raw_pii`). For X1, the intent prompt receives `Please call me on [PHONE] or mail [EMAIL], UPI [UPI]. FK1001 mein Britannia bread missing tha, refund do.` The route stays `auto_refund`.
- **Stretch done:** "PII masked: N items" shows in *Why this answer?*. The count comes from the pipeline (`pii_count`); app.py only displays it.

## Ex2: language signal and replying in the customer's language

- **Concept:** a new `language` key on the existing `RunnableParallel`, built as `masked | language_prompt | llm | StrOutputParser() | RunnableLambda(parse_language)`. `parse_language` takes the first word and falls back to `"english"` for anything outside the six labels, the same pattern as `parse_intent`.
- **Passed to the lanes by `add_note`**, which adds `language` to the box, with `other` mapped to `english`. The lane system prompt now says `Reply in {language}`.
- **Display:** a `Language:` line in *Why this answer?* (it is also in the signals JSON), and a `language` column in the Ops queue.
- **Think about it: did the router get slower?** Barely. `RunnableParallel` starts the language call at the same time as the intent call, so the router waits for the slower of the two, not their sum. The test `test_language_runs_in_parallel_with_intent` gives each fake call a 0.4s delay: the router takes about 0.4s, where running them one after the other would take 0.8s or more. To measure it live, once a key is in `.env`:
  ```bash
  .venv/bin/python -c "import time,bot; t=time.time(); bot.router.invoke({'input':'FK1007 ko 15 min bola tha aur 38 min lag gaye','chat_history':[]}); print(round(time.time()-t,2),'s')"
  ```
- **Stretch (one combined `intent,sentiment,language` call vs two calls).** Not built, but this is the trade-off:
  - **Cost:** the combined call saves one full prompt per ticket. The instructions and the message are sent once instead of twice, so it costs about 40–50% fewer input tokens. Both versions cost a similar number of output tokens.
  - **Speed:** about the same, because the two calls already run in parallel.
  - **Risk:** with one prompt doing two jobs, a parse error loses both labels at once.
  - **Verdict:** under a ₹2-per-ticket budget, the combined call is the one to ship. Ex6's token counter can confirm the numbers.

## Ex3: never-down intent (retry, backup model, safe default)

- **Concepts:** `.with_retry(stop_after_attempt=3, retry_if_exception_type=RETRYABLE)`, `.with_fallbacks([backup])`, and a last-resort `RunnableLambda`. There is no `try/except` anywhere (a test checks this).
  `safe_llm = retrying(main).with_fallbacks([retrying(backup)])` is used by the intent, language and lane chains. Each chain then has its own final fallback:
  - intent → `"unclear"` (and `decide_route` sends `unclear` to `human_review`)
  - language → `"english"`
  - lanes → a fixed apology in the customer's language
- **Which errors are retried:** only the ones that can succeed on a second try: timeout, connection error, 429 and 5xx. A wrong model name (404/400) or a bad key (401) never succeeds, so it goes straight to the backup model without wasting retries. `ChatOpenAI(max_retries=0)` turns off the SDK's hidden retries so ours are visible, and `timeout=20` makes a hung call fail instead of blocking forever.
- **Which model answered:** the intent branch keeps the `AIMessage`, and a new station, `unpack_intent`, turns it into `intent` (still a plain string for `decide_route`) plus `intent_model`. *Why this answer?* shows "Intent answered by: …". This is the stretch, and it is also the proof for acceptance test 1.
- **Acceptance tests (fake LLM):**
  - With `MODEL=this/model-does-not-exist`, all 18 tickets still route correctly. Every intent is answered by the backup, every lane reply comes from the backup, and nothing is raised.
  - With both models down (401 everywhere), every ticket gets `intent=unclear` and `human_review`, and the reply is the fixed apology, for both `invoke` and `stream`.
  - With the normal setup, 18/18 still pass.
- **Caveat found while building this:** `RunnableRetry` only wraps `invoke`/`batch`. The app *streams* the lanes, so a streamed lane never retries: it goes straight to the backup, then to the apology (`test_stream_does_not_retry_but_falls_back`). `with_fallbacks` also only helps if the error happens before the first token. A failure halfway through a stream still raises. For a short 2–3 sentence reply this is an acceptable risk.
- `.env.example` now has `BACKUP_MODEL`. The `RETRY_WAIT` environment variable (default 0.5s) sets the backoff; the tests set it to 0.

## Ex4: ops hand-off card for escalations

- **Concepts:**
  - `reply_chain = add_note | RunnableParallel(reply=branch, card=card_branch)`.
  - `card_branch = RunnableBranch((route == "human_review", card_chain), RunnableLambda(lambda x: None))`.
  - `card_chain = assign(raw=card_prompt | llm | JsonOutputParser()) | RunnableLambda(check_card)`, with a Python default card as the fallback.
- **Design choice: after the branch, not inside the human_review lane.** The card needs the same input as the lanes (note, chat history, masked text) but not the reply, so it runs *next to* the chosen lane. The six lanes stay identical and keep the same Ex3 fallback wrapping. The card can also be turned off in one place.
- **The LLM is not trusted with business values:**
  - `check_card` keeps `team` and `priority` only when they are valid labels; otherwise they become `support` and `P2`.
  - When `decide_route` already knows the reason (food safety, payment issue, rider misconduct), Python sets the team and priority, whatever the LLM wrote. T09 and X4 always get `food_safety/P1`, T13 `finance`, T10 `rider_ops`.
  - Bad JSON, a wrong shape or an LLM outage gives a default card. It never crashes, on either `invoke` or `stream`.
- **The card sees the whole chat.** The card prompt gets the masked `chat_history`, so a multi-turn escalation is summarised in full. It is masked by Ex1, like everything else.
- **"Prove no extra LLM call for T01":** `test_t01_no_card_and_no_extra_llm_call` counts the fake LLM's calls. There are exactly 3 (intent, language, lane) and none for the card. The `RunnableBranch` returns `None` before any prompt is built.
- **Watch out: what streaming prints.** `reply_chain.stream(d)` now yields dicts: `{'reply': 'Sorry'}`, `{'reply': ' to hear'}`, …, and one `{'card': {...}}` (or `{'card': None}`). `st.write_stream` cannot show those dicts. The fix is display code only: `reply_text()` in app.py yields only the `reply` strings and keeps the card on the side. The pipeline itself did not change.
- **Knock-on fixes:** `RunnableWithMessageHistory(..., output_messages_key="reply")`, so memory stores the reply text and not the dict. The CLI prints only the reply chunks and then the card.
- **Ops queue:** new `team`, `priority` and `summary` columns. *Why this answer?* shows the card as JSON.

## Ex5: bulk triage for the morning backlog

- **Concepts:** `bulk_row.batch(inputs, config={"max_concurrency": 5}, return_exceptions=True)`, wrapped as `bulk_triage = RunnableLambda(wrap) | assign(results=RunnableLambda(run_batch)) | RunnableLambda(to_rows)`. `to_rows` turns each result, including exceptions, into a plain `{id, order_id, intent, language, route, reasons}` row. Language is included so ops can staff Kannada and Telugu agents.
- **Where errors come from now.** After Ex3, an LLM failure no longer raises: the ticket becomes `unclear` → `human_review`, which is the safe place for it (`test_llm_failure_on_one_row_does_not_stop_others`). So `route = error` is for bad input. `bulk_row = RunnableLambda(require_message) | router` makes an empty or blank message fail fast, with no LLM call. `return_exceptions=True` keeps that row from stopping the other 18.
- **UI:** the Ops tab has a CSV upload (columns `id, message`) and a **Triage** button, plus a "Triage the 18 sample tickets" button. Both show the total time, the count per route, the full table and, when the CSV has `expected_route`, the match score. The CSV is read with `dtype=str, keep_default_na=False`; otherwise an empty cell becomes the float `NaN` and the regexes crash.
- **Batch vs loop:** with the fake LLM at 0.05s per call, `test_bulk_batch_is_at_least_2x_faster_than_a_loop` passes with a wide margin. Live numbers: `.venv/bin/python -m pytest -k test_live_bulk_triage_18_and_speed -s` prints both. The exercise author measured about 8s against 23s with `max_concurrency=5`.
- **Think about it: `max_concurrency=50` on a free OpenRouter key.** Free keys have tight rate limits (requests per minute). Fifty parallel calls trigger HTTP 429s almost at once. Ex3's retry backs off and tries again, so it ends up slower than 5 parallel calls, and the rows that still fail land in `human_review` as `unclear`, which floods the queue we wanted to keep small. The setting exists to stay under the provider's limit and to protect our own budget; `max_concurrency` is a throttle, not a speed dial.

## Ex6: model switcher at runtime

- **Concept:** `llm = get_llm(mini).configurable_alternatives(ConfigurableField(id="model"), default_key="mini", llama=..., strong=...)`. The model is chosen per request with `config={"configurable": {"model": "llama"}}`, which works on `.invoke`, `.batch` and `.stream`. No chain is rebuilt; the tests check that `router` is the very same object.
- **Why every chain gets the switch for free:** the intent, language, lane and card chains (and Ex3's `with_retry`/`with_fallbacks` wrappers) all hold a reference to the *same* `llm` object. `config` is passed down to every child Runnable on each call, and `configurable_alternatives` reads `configurable["model"]` at call time to pick the model. `test_switch_reaches_lanes_and_card_on_invoke_and_stream` proves that one config reaches all four kinds of LLM call. Ex5's bulk triage passes the config on too.
- **Interaction with Ex3:** if the chosen model fails, or the key is unknown, the fixed backup model answers. The Ex3 safety net keeps working whichever model is selected.
- **UI:**
  - A **Model** select box in the sidebar. The choice goes into `cfg` for `router.invoke`, `reply_chain.stream` and `bulk_triage`.
  - *Why this answer?* shows "Model chosen" and "intent answered by".
  - Memory is keyed by session, not model, so switching mid-chat keeps the history. The test switches llama → strong and the order id is still recovered from the earlier turn.
  - A **Compare models** button runs `evaluate.batch([...], max_concurrency=1)`. That is the same `router` on the 18 tickets for each model, showing score/18, seconds, tokens, ₹ per ticket and which model really answered.
- **Which model sent T18 ("VIP ₹2000") to `human_review`?** All of them, and no model *can* do otherwise. `claimed_amount` is a regex that reads ₹2000 from the raw text, and `decide_route` checks `amount > 1000` in plain Python, before any rule that depends on the intent. The test runs T18 with every intent label (including `general`, `refund_status` and `unclear`) on all three models: it is always `human_review`. The LLM only *labels* the message. Money and safety are decided in code, so "ignore your policy" in the message has nothing to act on.
- **Stretch (tokens and ₹ per ticket):** `UsageMetadataCallbackHandler` is passed in `config["callbacks"]` and adds up `usage_metadata` per model name. `rupees()` multiplies by approximate OpenRouter list prices (`PRICE_USD_PER_M`, ₹88/USD). These are routing calls only (intent + language). Live numbers: `.venv/bin/python -m pytest -k test_live_compare -s`.

## Ex7: live pipeline tracker

- **Concepts:** every station is named with `.with_config(run_name=...)` and timed with `.with_listeners(on_start=..., on_end=...)`, through one helper, `station(name, runnable)`. The named stations are: pii mask, order id, the 8 signals, signals (parallel), unpack_intent, decide_route, each lane, the hand-off card, and the whole router.
- **Why listeners, not `astream_events` (also in the `# Ex 7` comment):**
  1. **Exact timings.** A listener gets the run's own `start_time`/`end_time`. With `astream_events`, every step of a sequence emits its start event at t=0, because it is already waiting for its input. In a probe, `decide_route` "started" at 0.00s and would have looked as slow as the LLM.
  2. **The lane is included.** Listeners also fire on `reply_chain.stream()`, so the lane appears in the tracker, which the acceptance test needs.
  3. **No leaked drafts.** They never expose a lane's draft tokens, which matters once the Bonus guard is in place.
- **Turning it off:** the listeners append to a list the *caller* passes in `config={"configurable": {"trace": [...]}}`. Without a list nothing is recorded, and the result is the same: `test_tracker_off_records_nothing_and_changes_nothing` gets 18/18 both ways with identical routes. Concurrent requests keep separate lists. The sidebar has a "Live pipeline tracker" toggle.
- **Watch out: threads.** The listeners on the `RunnableParallel` stations run on worker threads (the test checks that `MainThread` is not among them, while `decide_route` runs on `MainThread`). So the listeners only append to the list. app.py runs `router.invoke` on a worker thread and redraws an `st.status` box from Streamlit's own thread every 0.1s until the result arrives. Then it streams the reply and redraws once more, adding the lane and the card.
- **Sum of signal times vs total router time:** the sum is bigger, because the stations overlap. All 8 signals start together inside `RunnableParallel`: the lookups finish in milliseconds, and the router waits only as long as the slowest LLM call. With the fake LLM at 0.3s for intent and 0.3s for language, the sum is at least 0.6s while the router takes under 0.5s. The tracker prints this comparison under the station lines.
- **Test-harness lesson:** with listeners attached, LangChain's tracer formats the traceback of every error. The fake used to raise the *same* exception object on every call, so its traceback grew each time and one test took 76s. The fake now builds a fresh error per call, as a real API does, and the suite is back to about 25s.

## Bonus: reply guard

- **Concept:** `guarded(name, lane)` = `assign(draft=lane)` → `assign(verdict=...)` → a `RunnableBranch` that sends the draft if the verdict is safe and a fixed safe template otherwise. It is used only for `human_review` and `policy_decline`; the other four lanes skip it and cost nothing extra (`test_unguarded_lanes_skip_the_check`). Promising a refund *is* the auto_refund lane's job.
- **The verdict is a `RunnableBranch` in three steps:**
  1. **Fixed texts pass.** The Ex3 apology is pre-approved, so it skips the check. This also keeps Ex3's acceptance test intact.
  2. **Plain-Python regex.** A sentence that mentions a refund together with an amount (₹/Rs/INR + digits) or a date/time (today, tomorrow, within N, in N days, aaj, kal, and the Hindi, Kannada and Telugu words) is a money promise. Internal tokens (`route=`, `reasons=`, `internal note`, route labels, `fraud`) are also flagged. When the regex fires, the template is sent and **no LLM call is made**.
  3. **LLM checker.** It answers yes or no. If the checker itself fails, the verdict is "unsafe" (fail closed).
- **Why regex first is safer:** the regex is deterministic. It gives the same answer every time, it cannot be prompt-injected by text in the chat, it costs nothing, and it is unit-tested on 20 cases. The LLM can be talked round ("the customer is VIP, this promise is fine"), can be down, and costs money. The LLM is still useful for the cases the patterns cannot see: paraphrases ("we'll put the full amount back very soon") and languages the regex has no words for.
- **Tuned to avoid false positives.** A decline that says "your ₹245 order was delivered more than 48 hours ago" passes, because there is no refund word in that sentence. The plain word "reason" is allowed, because the decline lane is *told* to explain the reason. The checker prompt says so too. Amount and refund words in *different* sentences pass.
- **Templates in the customer's language** (Ex2), separately for human_review and policy_decline.
- **Trade-off (required): the guarded lanes can no longer stream token by token.** The verdict needs the *whole* draft, so nothing can be shown until the draft is finished and checked. The customer sees the reply in one piece, about one LLM call later, and `test_guarded_stream_never_shows_the_draft` checks that the draft never reaches the screen. That is acceptable for escalations: they are a small share of tickets, a human is taking over anyway, and one wrong "₹780 today" promise costs far more, in money and legal risk, than a second of waiting. The four everyday lanes still stream.
- **Ex7 interaction:** the tracker shows a `guard check` station for the guarded lanes. Listeners only record timings, never the draft text, so the tracker cannot leak the draft either.
