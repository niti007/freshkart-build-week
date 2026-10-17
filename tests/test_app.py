import csv, importlib, os, sys, time
from pathlib import Path
import httpx, openai, pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, ConfigurableField
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
HAS_KEY = bool(os.getenv("OPENROUTER_API_KEY"))
TICKETS = list(csv.DictReader(open(ROOT / "data" / "tickets.csv", encoding="utf-8")))
BY_ID = {t["id"]: t for t in TICKETS}
BY_MSG = {t["message"]: t["expected_intent"] for t in TICKETS}
live = pytest.mark.skipif(not HAS_KEY, reason="no OPENROUTER_API_KEY")

# Extra test tickets from EXERCISES.md
X = {
    "X1": "Please call me on 98450 12345 or mail ravi.k@gmail.com, UPI ravi@okaxis. FK1001 mein Britannia bread missing tha, refund do.",
    "X2": "FK1007 ಆರ್ಡರ್ ತುಂಬಾ ತಡವಾಗಿ ಬಂತು, ಯಾಕೆ?",
    "X3": "FK1003 లో వచ్చిన పెరుగు ఎక్స్‌పైర్ అయిపోయింది, రీఫండ్ కావాలి",
    "X4": "FK1009 ka khana kha ke bacche ko ulti ho rahi hai",
    "X5": "",
}
# Messages that are not in tickets.csv (or arrive masked) are matched by a fragment.
INTENT_HINTS = [("Britannia bread missing", "missing_item"), ("ತಡವಾಗಿ", "wismo"), ("ఎక్స్‌పైర్", "expired"),
                ("ulti", "food_safety")]

# ---- fake LLM ----
# One shared state dict, reset by the fixtures. Tests read and tweak it:
#   calls  every LLM call: model, kind (intent/language/card/checker/lane), system, human, messages
#   fail   model (or "*") -> an exception, or a list of exceptions raised one per call until empty
#   script kind -> function(call) -> text, to override the canned answer
#   delay  kind -> seconds to sleep (timing tests)
FAKE = {"calls": [], "fail": {}, "script": {}, "delay": {}}

def reset_fake():
    FAKE.update(calls=[], fail={}, script={}, delay={})

def prompt_kind(system):
    if "Classify" in system: return "intent"
    if "Which language" in system: return "language"
    if "team, priority" in system: return "card"
    if "Answer yes or no" in system: return "checker"
    return "lane"

def fake_intent(call):
    human = call["human"]
    if human in BY_MSG: return BY_MSG[human] + ",neutral"
    return next((i for frag, i in INTENT_HINTS if frag in human), "general") + ",neutral"

DEFAULTS = {"intent": fake_intent, "lane": lambda call: "Fake reply.", "language": lambda call: "english",
            "card": lambda call: '{"team": "support", "priority": "P3", "summary_en": "Fake summary.", "next_step": "Fake step."}',
            "checker": lambda call: "no"}

def fake_llm(prompt_value, model):
    msgs = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else [HumanMessage(content=prompt_value)]
    call = {"model": model, "kind": prompt_kind(msgs[0].content), "system": msgs[0].content,
            "human": msgs[-1].content, "messages": msgs}
    FAKE["calls"].append(call)
    time.sleep(FAKE["delay"].get(call["kind"], 0))
    err = FAKE["fail"].get(model, FAKE["fail"].get("*"))
    if isinstance(err, list) and err: raise err.pop(0)
    if isinstance(err, Exception): raise err
    text = FAKE["script"].get(call["kind"], DEFAULTS[call["kind"]])(call)
    return AIMessage(content=text, response_metadata={"model_name": model},
                     usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})

def fake_chat_openai(**kw):
    """Stands in for ChatOpenAI. A RunnableSequence (not a bare RunnableLambda) so it has configurable_alternatives."""
    model = kw.get("model")
    return RunnablePassthrough() | RunnableLambda(lambda pv: fake_llm(pv, model))

def calls(kind=None):
    return [c for c in FAKE["calls"] if kind is None or c["kind"] == kind]

# openai errors the way the SDK raises them
REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
def status_error(cls, code): return cls(f"HTTP {code}", response=httpx.Response(code, request=REQ), body=None)
def timeout_error(): return openai.APITimeoutError(request=REQ)
def not_found_error(): return status_error(openai.NotFoundError, 404)
def auth_error(): return status_error(openai.AuthenticationError, 401)
def server_error(): return status_error(openai.InternalServerError, 500)

def load_bot(monkeypatch, fake):
    import langchain_openai
    reset_fake()
    if fake:
        monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)
        monkeypatch.setenv("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "test-key"))
    sys.modules.pop("bot", None)
    return importlib.import_module("bot")

@pytest.fixture
def fake_bot(monkeypatch):
    yield load_bot(monkeypatch, fake=True)
    sys.modules.pop("bot", None)

@pytest.fixture
def real_bot(monkeypatch):
    yield load_bot(monkeypatch, fake=False)
    sys.modules.pop("bot", None)

def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

def inputs(tickets=TICKETS):
    return [{"input": t["message"], "chat_history": []} for t in tickets]

def wrong_routes(results, tickets=TICKETS):
    """The regression check used after every exercise: ids of tickets whose route is not the expected one."""
    return [t["id"] for t, r in zip(tickets, results) if r["route"] != t["expected_route"]]

def decide(bot, msg, history=()):
    return bot.router.invoke({"input": msg, "chat_history": list(history)})

# ---- Phase 0: the fake LLM itself ----
def test_fake_answers_every_ticket_intent(fake_bot):
    for t in TICKETS:
        decide(fake_bot, t["message"])
    assert [c["human"] for c in calls("intent")] == [t["message"] for t in TICKETS]
    assert all(c["model"] == "openai/gpt-4o-mini" for c in calls())

def test_fake_returns_ai_message_with_metadata():
    out = fake_chat_openai(model="m").invoke("Classify this")
    assert isinstance(out, AIMessage)
    assert out.response_metadata["model_name"] == "m" and out.usage_metadata["total_tokens"] == 110

def test_fake_failure_injection_list_then_success():
    reset_fake()
    FAKE["fail"]["m"] = [timeout_error(), timeout_error()]
    llm = fake_chat_openai(model="m")
    for _ in range(2):
        with pytest.raises(openai.APITimeoutError): llm.invoke("hi")
    assert llm.invoke("hi").content == "Fake reply."

def test_fake_failure_by_model_and_wildcard():
    reset_fake()
    FAKE["fail"]["*"] = auth_error()
    with pytest.raises(openai.AuthenticationError): fake_chat_openai(model="any").invoke("hi")
    FAKE["fail"]["ok"] = None
    assert fake_chat_openai(model="ok").invoke("hi").content == "Fake reply."

def test_fake_supports_configurable_alternatives():
    reset_fake()
    llm = fake_chat_openai(model="a").configurable_alternatives(ConfigurableField(id="model"), default_key="a",
                                                                  b=fake_chat_openai(model="b"))
    llm.invoke("hi"); llm.invoke("hi", config={"configurable": {"model": "b"}})
    assert [c["model"] for c in calls()] == ["a", "b"]

def test_fake_script_and_delay():
    reset_fake()
    FAKE["script"]["lane"] = lambda call: "scripted"
    FAKE["delay"]["lane"] = 0.05
    t0 = time.perf_counter()
    assert fake_chat_openai(model="m").invoke("hi").content == "scripted"
    assert time.perf_counter() - t0 >= 0.05

# ---- baseline: helpers and rules (regression net before any exercise) ----
@pytest.mark.parametrize("text,oid", [
    ("FK1001 mein bread", "FK1001"), ("fk1006 kab tak", "FK1006"), ("ord_1010 ka rider", "FK1010"),
    ("order FK 1002 please", "FK1002"), ("Mera order abhi tak nahi aaya", None), ("FK12345", None),
])
def test_find_order_id(fake_bot, text, oid):
    assert fake_bot.find_order_id(text) == oid

@pytest.mark.parametrize("text,amount", [
    ("₹520 twice debited", 520.0), ("Rs. 1,200 refund", 1200.0), ("INR 300", 300.0), ("Rs 50 and ₹2000", 2000.0),
    ("no money here, FK1001", None),
])
def test_claimed_amount(fake_bot, text, amount):
    assert fake_bot.claimed_amount(text) == amount

@pytest.mark.parametrize("text,flag", [
    ("meri beti ko ulti ho rahi hai", True), ("Possibly food poisoning", True), ("baby has a rash", True),
    ("mould on bread", True), ("bread missing tha", False), ("rider was rude", False),
])
def test_has_safety(fake_bot, text, flag):
    assert fake_bot.has_safety(text) is flag

def test_lookups(fake_bot):
    assert fake_bot.get_order("FK1001")["amount"] == "210"
    assert fake_bot.get_order("FK9999") is None and fake_bot.get_customer("FK9999") is None
    assert fake_bot.check_inventory("FK1003") == {"expired_confirmed": True, "baby": False}
    assert fake_bot.check_inventory("FK1004")["baby"] is True
    assert fake_bot.check_inventory(None) is None
    assert fake_bot.get_rider("FK1014")["gps_delivered"] == "n"

def test_order_id_prefers_current_message_then_latest_history(fake_bot):
    hist = [HumanMessage(content="FK1001 ka status?"), AIMessage(content="ok"), HumanMessage(content="FK1016 bhi")]
    assert fake_bot.order_id_from({"input": "FK1003 ka kya?", "chat_history": hist}) == "FK1003"
    assert fake_bot.order_id_from({"input": "refund kab?", "chat_history": hist}) == "FK1016"
    assert fake_bot.order_id_from({"input": "hello"}) is None

def test_add_note_mentions_coupon_only_when_late(fake_bot):
    late = fake_bot.add_note(decide(fake_bot, BY_ID["T07"]["message"]))["note"]
    on_time = fake_bot.add_note(decide(fake_bot, BY_ID["T01"]["message"]))["note"]
    assert "delay=23min" in late and "Rs 50 coupon" in late
    assert "coupon" not in on_time and "order=FK1001" in on_time

def test_router_output_shape(fake_bot):
    d = decide(fake_bot, BY_ID["T01"]["message"])
    assert {"input", "chat_history", "order_id", "signals", "route", "reasons"} <= set(d)
    assert set(d["signals"]) >= {"intent", "order", "customer", "inventory", "rider", "safety", "claimed_amount"}

@pytest.mark.parametrize("tid", [t["id"] for t in TICKETS])
def test_each_route_uses_its_lane(fake_bot, tid):
    d = decide(fake_bot, BY_ID[tid]["message"])
    assert fake_bot.reply_chain.invoke(d) == "Fake reply."
    assert fake_bot.LANES[d["route"]] in calls("lane")[-1]["system"]

# ---- offline: rules and router ----
CLEAN = {"customer_id": "C01", "fraud_score": "0.05", "refunds_30d": "0"}
BASE = {"intent": "missing_item",
        "order": {"amount": "210", "status": "delivered", "delivered_at": "2026-10-12 18:40", "substituted": "n"},
        "customer": CLEAN, "inventory": {"expired_confirmed": False, "baby": False},
        "rider": {"gps_delivered": "y"}, "safety": False, "claimed_amount": None}

@pytest.mark.parametrize("change,route", [
    ({}, "auto_refund"),
    ({"claimed_amount": 2000.0}, "human_review"),
    ({"inventory": {"expired_confirmed": False, "baby": True}}, "human_review"),
    ({"safety": True}, "human_review"),
    ({"intent": "general"}, "info"),
    ({"order": None}, "info"),
    ({"intent": "mystery"}, "human_review"),
    ({"intent": "unclear"}, "human_review"),
    ({"intent": "payment_issue"}, "human_review"),
    ({"intent": "rider_misconduct"}, "human_review"),
    ({"rider": {"gps_delivered": "n"}}, "human_review"),
    ({"customer": {**CLEAN, "fraud_score": "0.9"}}, "human_review"),
    ({"order": {**BASE["order"], "status": "packing"}}, "policy_decline"),
    ({"order": {**BASE["order"], "delivered_at": "2026-10-09 10:00"}}, "policy_decline"),
    ({"intent": "expired"}, "policy_decline"),
    ({"intent": "wrong_item_substitution", "order": {**BASE["order"], "substituted": "y"}}, "substitution"),
    ({"claimed_amount": 500.0}, "human_review"),
])
def test_decide_route(fake_bot, change, route):
    assert fake_bot.decide_route({"signals": {**BASE, **change}})["route"] == route

def test_router_all_18_tickets(fake_bot):
    results = fake_bot.router.batch(inputs())
    assert len(TICKETS) == 18 and not wrong_routes(results), wrong_routes(results)

def test_memory_recovers_order_id(fake_bot):
    x = {"input": "refund kab milega?", "chat_history": [HumanMessage(content="FK1016 ka status?")]}
    assert fake_bot.order_id_from(x) == "FK1016"

# ---- UI with a fake LLM ----
def test_app_loads(fake_bot):
    at = app()
    assert not at.exception
    assert at.title[0].value == "FreshKart Support Desk"
    assert len(at.tabs) == 4 and len(at.dataframe) == 4  # Data tab: 4 CSVs

def test_chat_roundtrip(fake_bot):
    at = app()
    at.chat_input[0].set_value(TICKETS[0]["message"]).run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert at.expander[0].label == "Why this answer?"
    assert "auto_refund" in at.expander[0].markdown[0].value
    assert "Fake reply." in at.chat_message[1].markdown[0].value
    assert at.session_state["messages"][1]["route"] == "auto_refund"

def test_ops_queue_and_new_chat(fake_bot):
    at = app()
    at.chat_input[0].set_value(TICKETS[8]["message"]).run()  # T09 food safety
    assert at.session_state["log"][0]["route"] == "human_review"
    assert any(m.label == "human_review" and m.value == "1" for m in at.metric)
    assert len(at.dataframe) == 5  # queue + 4 data tables
    next(b for b in at.sidebar.button if b.label == "New chat").click().run()
    assert at.session_state["messages"] == [] and len(at.chat_message) == 0

def test_sample_ticket_send(fake_bot):
    at = app()
    next(b for b in at.sidebar.button if b.label == "Send").click().run()
    assert len(at.chat_message) == 2

def test_missing_key_shows_error(monkeypatch, fake_bot):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    at = app()
    assert at.error and "OPENROUTER_API_KEY" in at.error[0].value

# ---- Ex1: PII shield ----
PII_RAW = ["98450 12345", "ravi.k@gmail.com", "ravi@okaxis"]

@pytest.mark.parametrize("text,masked,count", [
    (X["X1"], "Please call me on [PHONE] or mail [EMAIL], UPI [UPI]. FK1001 mein Britannia bread missing tha, refund do.", 3),
    ("call +91 98450 12345 now", "call [PHONE] now", 1),
    ("call +91-9845012345", "call [PHONE]", 1),
    ("my number is 9845012345.", "my number is [PHONE].", 1),
    ("card 4111 1111 1111 1111 please", "card [CARD] please", 1),
    ("card 4111-1111-1111-1111", "card [CARD]", 1),
    ("card 6011111111111117", "card [CARD]", 1),
    ("aadhaar-like 1234 5678 9012", "aadhaar-like [CARD]", 1),
    ("mail me at first.last+fk@mail.co.in.", "mail me at [EMAIL].", 1),
    ("pay to 9845012345@paytm", "pay to [UPI]", 1),
    ("upi: name.k@ybl", "upi: [UPI]", 1),
    (BY_ID["T13"]["message"], BY_ID["T13"]["message"], 0),
    ("FK1001 ord_1010 FK 1002 ₹520 Rs. 1,200 INR 300", "FK1001 ord_1010 FK 1002 ₹520 Rs. 1,200 INR 300", 0),
    ("delivered 2026-10-12 18:40, 38 min late, 11 Oct", "delivered 2026-10-12 18:40, 38 min late, 11 Oct", 0),
    ("short 12345 and 98450", "short 12345 and 98450", 0),
    ("", "", 0),
])
def test_mask_pii(fake_bot, text, masked, count):
    assert fake_bot.mask_pii(text) == (masked, count)

def test_x1_intent_prompt_is_masked_but_routing_unchanged(fake_bot):
    d = decide(fake_bot, X["X1"])
    seen = calls("intent")[-1]["human"]
    assert "[PHONE]" in seen and "[EMAIL]" in seen and "[UPI]" in seen and "FK1001" in seen
    assert not any(p in seen for p in PII_RAW)
    assert d["order_id"] == "FK1001" and d["route"] == "auto_refund"
    assert d["input"] == X["X1"] and d["pii_count"] == 3

def test_x1_intent_prompt_text_via_to_string(fake_bot):
    d = fake_bot.mask.invoke({"input": X["X1"], "chat_history": []})
    shown = fake_bot.intent_prompt.invoke(fake_bot.masked.invoke(d)).to_string()
    assert "[PHONE]" in shown and "98450" not in shown

def test_no_llm_call_ever_sees_raw_pii(fake_bot):
    d = decide(fake_bot, X["X1"])
    fake_bot.reply_chain.invoke(d)
    text = " ".join(m.content for c in calls() for m in c["messages"])
    assert calls("lane") and not any(p in text for p in PII_RAW)

def test_lane_prompt_reads_masked_input(fake_bot):
    fake_bot.reply_chain.invoke(decide(fake_bot, X["X1"]))
    assert calls("lane")[-1]["human"].startswith("Please call me on [PHONE]")

def test_history_is_masked_but_order_id_still_recovered(fake_bot):
    hist = [HumanMessage(content="FK1016 ka refund? call 9845012345"), AIMessage(content="Checking.")]
    d = decide(fake_bot, "refund kab milega?", hist)
    assert d["order_id"] == "FK1016"
    assert "9845012345" not in d["chat_history"][0].content and "[PHONE]" in d["chat_history"][0].content
    fake_bot.reply_chain.invoke(d)
    assert "9845012345" not in " ".join(m.content for m in calls("lane")[-1]["messages"])
    assert hist[0].content.endswith("9845012345")  # the caller's history object is not modified

def test_t13_amount_not_masked(fake_bot):
    d = decide(fake_bot, BY_ID["T13"]["message"])
    assert d["signals"]["claimed_amount"] == 520.0 and d["masked_input"] == d["input"] and d["pii_count"] == 0

def test_regex_signals_read_original_input(fake_bot):
    d = decide(fake_bot, "FK1001 bread missing, ₹1500 refund do. Call 9845012345, bacche ko ulti ho rahi hai")
    assert d["signals"]["claimed_amount"] == 1500.0 and d["signals"]["safety"] is True
    assert d["route"] == "human_review"

def test_ui_shows_pii_badge(fake_bot):
    at = app()
    at.chat_input[0].set_value(X["X1"]).run()
    assert not at.exception
    assert any("PII masked: **3 items**" in md.value for md in at.expander[0].markdown)
    assert at.session_state["messages"][1]["pii_count"] == 3

def test_ui_no_pii_badge_without_pii(fake_bot):
    at = app()
    at.chat_input[0].set_value(BY_ID["T01"]["message"]).run()
    assert not any("PII masked" in md.value for md in at.expander[0].markdown)

@live
def test_live_x1_masked_and_auto_refund(real_bot):
    d = decide(real_bot, X["X1"])
    assert d["route"] == "auto_refund" and d["order_id"] == "FK1001" and d["pii_count"] == 3

# ---- live (real OpenRouter calls) ----
@live
@pytest.mark.parametrize("tid,route", [("T09", "human_review"), ("T01", "auto_refund")])
def test_live_round_trip(real_bot, tid, route):
    msg = next(t["message"] for t in TICKETS if t["id"] == tid)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.chat_input[0].set_value(msg).run()
    assert not at.exception
    last = at.session_state["messages"][-1]
    assert last["route"] == route and last["content"].strip()

@live
def test_live_router_18_of_18(real_bot):
    results = real_bot.router.batch(inputs(), config={"max_concurrency": 5})
    assert not wrong_routes(results), wrong_routes(results)
