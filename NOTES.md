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
