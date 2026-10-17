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
