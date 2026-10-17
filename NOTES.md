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
