import csv, importlib, os, re, sys, time
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
#   init   kwargs of every ChatOpenAI(...) the bot built
FAKE = {"calls": [], "fail": {}, "script": {}, "delay": {}, "init": []}

def reset_fake():
    FAKE.update(calls=[], fail={}, script={}, delay={}, init=[])

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

def fake_language(call):
    """Answers like a real model would: by script, then by common Hinglish words."""
    h = call["human"]
    if re.search("[\u0C80-\u0CFF]", h): return "kannada"
    if re.search("[\u0C00-\u0C7F]", h): return "telugu"
    if re.search("[\u0900-\u097F]", h): return "hindi"
    if re.search(r"(?i)\b(mein|ka|ke|ki|hai|tha|kar|nahi|do|chahiye|kab|bola|maine|aur|raha|gaya|mujhe)\b", h): return "hinglish"
    return "english"

DEFAULTS = {"intent": fake_intent, "lane": lambda call: "Fake reply.", "language": fake_language,
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
    FAKE["init"].append(kw)
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
        monkeypatch.setenv("RETRY_WAIT", "0")
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

def reply(bot, d):
    return bot.reply_chain.invoke(d)["reply"]

def streamed(bot, d):
    return "".join(chunk.get("reply", "") for chunk in bot.reply_chain.stream(d))

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
    assert reply(fake_bot, d) == "Fake reply."
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
    reply(fake_bot, d)
    text = " ".join(m.content for c in calls() for m in c["messages"])
    assert calls("lane") and not any(p in text for p in PII_RAW)

def test_lane_prompt_reads_masked_input(fake_bot):
    reply(fake_bot, decide(fake_bot, X["X1"]))
    assert calls("lane")[-1]["human"].startswith("Please call me on [PHONE]")

def test_history_is_masked_but_order_id_still_recovered(fake_bot):
    hist = [HumanMessage(content="FK1016 ka refund? call 9845012345"), AIMessage(content="Checking.")]
    d = decide(fake_bot, "refund kab milega?", hist)
    assert d["order_id"] == "FK1016"
    assert "9845012345" not in d["chat_history"][0].content and "[PHONE]" in d["chat_history"][0].content
    reply(fake_bot, d)
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

# ---- Ex2: language signal ----
@pytest.mark.parametrize("raw,label", [
    ("kannada", "kannada"), ("Kannada.", "kannada"), ("  TELUGU\n", "telugu"), ("hinglish", "hinglish"),
    ("Hindi", "hindi"), ("english", "english"), ("other", "other"), ("klingon", "english"), ("", "english"),
    ("The language is Kannada", "english"), ("123", "english"),
])
def test_parse_language(fake_bot, raw, label):
    assert fake_bot.parse_language(raw) == label

@pytest.mark.parametrize("msg,lang", [
    (X["X2"], "kannada"), (X["X3"], "telugu"), (BY_ID["T01"]["message"], "hinglish"),
    (BY_ID["T03"]["message"], "english"), (X["X4"], "hinglish"), ("मेरा ऑर्डर FK1006 कहाँ है?", "hindi"),
])
def test_language_signal(fake_bot, msg, lang):
    assert decide(fake_bot, msg)["signals"]["language"] == lang

def test_t01_is_hinglish(fake_bot):
    assert decide(fake_bot, BY_ID["T01"]["message"])["signals"]["language"] == "hinglish"

def test_language_falls_back_to_english_on_bad_answer(fake_bot):
    FAKE["script"]["language"] = lambda call: "Klingon, definitely"
    assert decide(fake_bot, X["X2"])["signals"]["language"] == "english"

def test_language_reads_masked_text(fake_bot):
    decide(fake_bot, X["X1"])
    assert "[PHONE]" in calls("language")[-1]["human"] and "98450" not in calls("language")[-1]["human"]

def test_x2_kannada_wismo_with_coupon(fake_bot):
    d = decide(fake_bot, X["X2"])
    assert d["route"] == "wismo" and d["signals"]["language"] == "kannada"
    noted = fake_bot.add_note(d)
    assert noted["language"] == "kannada" and "Rs 50 coupon" in noted["note"]
    reply(fake_bot, d)
    assert "Reply in kannada" in calls("lane")[-1]["system"] and "Rs 50 coupon" in calls("lane")[-1]["system"]

def test_x3_telugu_reply_language(fake_bot):
    d = decide(fake_bot, X["X3"])
    assert d["signals"]["language"] == "telugu" and d["order_id"] == "FK1003"
    reply(fake_bot, d)
    assert "Reply in telugu" in calls("lane")[-1]["system"]

def test_other_language_replies_in_english(fake_bot):
    FAKE["script"]["language"] = lambda call: "other"
    d = decide(fake_bot, "Bonjour, ma commande FK1006?")
    assert d["signals"]["language"] == "other" and fake_bot.add_note(d)["language"] == "english"
    reply(fake_bot, d)
    assert "Reply in english" in calls("lane")[-1]["system"]

def test_language_does_not_change_routes(fake_bot):
    FAKE["script"]["language"] = lambda call: "telugu"
    assert not wrong_routes(fake_bot.router.batch(inputs()))

def test_language_runs_in_parallel_with_intent(fake_bot):
    FAKE["delay"].update(intent=0.4, language=0.4)
    t0 = time.perf_counter()
    decide(fake_bot, BY_ID["T01"]["message"])
    elapsed = time.perf_counter() - t0
    assert len(calls("intent")) == 1 and len(calls("language")) == 1
    assert 0.4 <= elapsed < 0.7, elapsed  # in sequence it would be >= 0.8s

def test_ui_shows_language_and_ops_column(fake_bot):
    at = app()
    at.chat_input[0].set_value(X["X2"]).run()
    assert not at.exception
    assert any("Language: **kannada**" in md.value for md in at.expander[0].markdown)
    assert at.session_state["log"][0]["language"] == "kannada"

def test_ops_queue_has_language_column(fake_bot):
    at = app()
    at.chat_input[0].set_value(BY_ID["T09"]["message"]).run()
    assert "language" in at.dataframe[0].value.columns

KANNADA, TELUGU = re.compile("[ಀ-೿]"), re.compile("[ఀ-౿]")

@live
def test_live_x2_kannada_reply_with_coupon(real_bot):
    d = decide(real_bot, X["X2"])
    assert d["signals"]["language"] == "kannada" and d["route"] == "wismo"
    text = reply(real_bot, d)
    assert KANNADA.search(text) and "50" in text

@live
def test_live_x3_telugu_reply(real_bot):
    d = decide(real_bot, X["X3"])
    assert d["signals"]["language"] == "telugu" and TELUGU.search(reply(real_bot, d))

@live
def test_live_t01_hinglish(real_bot):
    assert decide(real_bot, BY_ID["T01"]["message"])["signals"]["language"] == "hinglish"

# ---- Ex3: retry + fallback model + safe default ----
MAIN, BACKUP, BROKEN = "openai/gpt-4o-mini", "meta-llama/llama-3.1-8b-instruct", "this/model-does-not-exist"

def boom(error):
    """A scripted answer that fails: lets one prompt kind fail while the others work."""
    def fail(call): raise error()
    return fail

@pytest.fixture
def broken_main_bot(monkeypatch):
    """.env has MODEL=this/model-does-not-exist (OpenRouter answers 404)."""
    monkeypatch.setenv("MODEL", BROKEN)
    bot = load_bot(monkeypatch, fake=True)
    FAKE["fail"][BROKEN] = not_found_error()
    yield bot
    sys.modules.pop("bot", None)

def test_llms_built_without_sdk_retries_and_with_timeout(fake_bot):
    assert {kw["model"] for kw in FAKE["init"]} == {MAIN, BACKUP}
    assert all(kw["max_retries"] == 0 and kw["timeout"] == 20 for kw in FAKE["init"])

def test_backup_model_from_env(monkeypatch):
    monkeypatch.setenv("BACKUP_MODEL", "some/other-model")
    load_bot(monkeypatch, fake=True)
    assert "some/other-model" in {kw["model"] for kw in FAKE["init"]}
    sys.modules.pop("bot", None)

@pytest.mark.parametrize("errors", [
    [timeout_error, timeout_error], [server_error], [lambda: status_error(openai.RateLimitError, 429)],
    [lambda: openai.APIConnectionError(request=REQ)],
])
def test_transient_errors_are_retried_on_main(fake_bot, errors):
    FAKE["fail"][MAIN] = [e() for e in errors]
    assert fake_bot.safe_llm.invoke("Classify: hi").content.startswith("general")
    assert [c["model"] for c in calls()] == [MAIN] * (len(errors) + 1)

def test_retries_exhausted_then_backup(fake_bot):
    FAKE["fail"][MAIN] = [timeout_error() for _ in range(3)]
    out = fake_bot.safe_llm.invoke("Classify: hi")
    assert [c["model"] for c in calls()] == [MAIN, MAIN, MAIN, BACKUP]
    assert out.response_metadata["model_name"] == BACKUP

@pytest.mark.parametrize("error", [not_found_error, auth_error, lambda: status_error(openai.BadRequestError, 400)])
def test_permanent_errors_skip_retry_and_go_to_backup(fake_bot, error):
    FAKE["fail"][MAIN] = error()
    fake_bot.safe_llm.invoke("Classify: hi")
    assert [c["model"] for c in calls()] == [MAIN, BACKUP]

def test_safe_llm_raises_only_when_both_models_fail(fake_bot):
    FAKE["fail"]["*"] = auth_error()
    with pytest.raises(openai.AuthenticationError):
        fake_bot.safe_llm.invoke("Classify: hi")

def test_unpack_intent(fake_bot):
    x = {"signals": {"intent_msg": AIMessage(content="food_safety,angry", response_metadata={"model_name": "m"}),
                     "safety": True}}
    s = fake_bot.unpack_intent(x)["signals"]
    assert s == {"safety": True, "intent": "food_safety", "intent_model": "m"}

def test_main_model_missing_all_18_still_answered_by_backup(broken_main_bot):
    results = broken_main_bot.router.batch(inputs())
    assert not wrong_routes(results)
    assert {r["signals"]["intent_model"] for r in results} == {BACKUP}
    for r in results:
        assert reply(broken_main_bot, r) == "Fake reply."
    assert [c["model"] for c in calls("lane")] == [BROKEN, BACKUP] * 18  # main tried once (404: no retry), then backup

def test_main_model_missing_stream_uses_backup(broken_main_bot):
    d = decide(broken_main_bot, BY_ID["T01"]["message"])
    assert streamed(broken_main_bot, d) == "Fake reply."

def test_both_models_down_unclear_human_review_apology(fake_bot):
    FAKE["fail"]["*"] = auth_error()
    results = fake_bot.router.batch(inputs())
    for r in results:
        assert r["signals"]["intent"] == "unclear" and r["signals"]["intent_model"] == "none (AI down)"
        assert r["signals"]["language"] == "english" and r["route"] == "human_review"
        assert reply(fake_bot, r) == fake_bot.APOLOGY["english"]
    assert streamed(fake_bot, results[0]) == fake_bot.APOLOGY["english"]

def test_both_down_unclear_reason(fake_bot):
    FAKE["fail"]["*"] = timeout_error()
    d = decide(fake_bot, BY_ID["T01"]["message"])  # no safety words, no amount: the unclear rule decides
    assert d["reasons"] == ["unclear intent"]

def test_apology_in_customer_language(fake_bot):
    FAKE["script"]["lane"] = boom(auth_error)
    d = decide(fake_bot, X["X2"])
    assert reply(fake_bot, d) == fake_bot.APOLOGY["kannada"]
    assert fake_bot.apology({"language": "other"}) == fake_bot.APOLOGY["english"]
    assert fake_bot.apology({}) == fake_bot.APOLOGY["english"]

def test_intent_falls_back_while_language_still_works(fake_bot):
    FAKE["script"]["intent"] = boom(auth_error)
    d = decide(fake_bot, X["X3"])
    assert d["signals"]["intent"] == "unclear" and d["signals"]["language"] == "telugu" and d["route"] == "human_review"

def test_stream_does_not_retry_but_falls_back(fake_bot):
    d = decide(fake_bot, BY_ID["T01"]["message"])
    FAKE["calls"].clear()
    FAKE["fail"][MAIN] = [timeout_error()]
    assert streamed(fake_bot, d) == "Fake reply."
    assert [c["model"] for c in calls("lane")] == [MAIN, BACKUP]  # documented: RunnableRetry does not wrap .stream

def test_invoke_retries_the_lane(fake_bot):
    d = decide(fake_bot, BY_ID["T01"]["message"])
    FAKE["calls"].clear()
    FAKE["fail"][MAIN] = [timeout_error()]
    assert reply(fake_bot, d) == "Fake reply."
    assert [c["model"] for c in calls("lane")] == [MAIN, MAIN]

def test_no_try_except_in_bot():
    src = (ROOT / "bot.py").read_text()
    assert not re.search(r"^\s*(try:|except\b)", src, re.M)

def test_ui_shows_backup_model(broken_main_bot):
    at = app()
    at.chat_input[0].set_value(BY_ID["T01"]["message"]).run()
    assert not at.exception
    assert any(f"Intent answered by: `{BACKUP}`" in md.value for md in at.expander[0].markdown)

def test_ui_never_crashes_when_ai_is_down(fake_bot):
    FAKE["fail"]["*"] = auth_error()
    at = app()
    at.chat_input[0].set_value(BY_ID["T01"]["message"]).run()
    assert not at.exception
    last = at.session_state["messages"][-1]
    assert last["route"] == "human_review" and last["content"] == fake_bot.APOLOGY["english"]

@live
def test_live_bad_main_model_uses_backup(monkeypatch):
    monkeypatch.setenv("MODEL", BROKEN)
    bot = load_bot(monkeypatch, fake=False)
    d = decide(bot, BY_ID["T01"]["message"])
    assert "llama" in d["signals"]["intent_model"] and d["route"] == "auto_refund"
    assert reply(bot, d).strip()
    sys.modules.pop("bot", None)

@live
def test_live_wrong_key_is_unclear_human_review_apology(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-wrong-key-for-this-test")
    bot = load_bot(monkeypatch, fake=False)
    d = decide(bot, BY_ID["T01"]["message"])
    assert d["signals"]["intent"] == "unclear" and d["route"] == "human_review"
    assert reply(bot, d) == bot.APOLOGY["english"]
    sys.modules.pop("bot", None)

# ---- Ex4: ops hand-off card ----
CARD_KEYS = {"team", "priority", "summary_en", "next_step"}

def card_of(bot, msg, history=()):
    return bot.reply_chain.invoke(decide(bot, msg, history))["card"]

@pytest.mark.parametrize("tid,team,priority", [
    ("T09", "food_safety", "P1"), ("T13", "finance", "P2"), ("T10", "rider_ops", "P2"),
])
def test_card_team_and_priority_for_known_reasons(fake_bot, tid, team, priority):
    card = card_of(fake_bot, BY_ID[tid]["message"])  # the fake LLM says support/P3: Python rules win
    assert card["team"] == team and card["priority"] == priority and set(card) == CARD_KEYS

def test_x4_hinglish_food_safety_p1(fake_bot):
    card = card_of(fake_bot, X["X4"])
    assert (card["team"], card["priority"]) == ("food_safety", "P1")
    assert "summary_en is one sentence in English" in calls("card")[-1]["system"]

def test_card_uses_llm_team_for_other_escalations(fake_bot):
    FAKE["script"]["card"] = lambda call: ('{"team": "finance", "priority": "P2", "summary_en": "Amount above limit.",'
                                           ' "next_step": "Check the order value."}')
    card = card_of(fake_bot, BY_ID["T18"]["message"])  # amount>1000: no fixed team
    assert card == {"team": "finance", "priority": "P2", "summary_en": "Amount above limit.", "next_step": "Check the order value."}

@pytest.mark.parametrize("raw,team,priority", [
    ('{"team": "aliens", "priority": "P9", "summary_en": "x", "next_step": "y"}', "support", "P2"),
    ('{"team": "rider_ops", "priority": "P1", "summary_en": "x", "next_step": "y"}', "rider_ops", "P1"),
    ('{"team": "FINANCE", "priority": "p2", "summary_en": "x", "next_step": "y"}', "support", "P2"),
])
def test_card_values_are_checked_in_python(fake_bot, raw, team, priority):
    FAKE["script"]["card"] = lambda call: raw
    card = card_of(fake_bot, BY_ID["T18"]["message"])
    assert (card["team"], card["priority"]) == (team, priority)

@pytest.mark.parametrize("raw", ["not json at all", "[1, 2, 3]", '"just a string"', '{"team": "finance"}'])
def test_bad_card_answers_never_crash(fake_bot, raw):
    FAKE["script"]["card"] = lambda call: raw
    d = decide(fake_bot, BY_ID["T09"]["message"])
    for card in [fake_bot.reply_chain.invoke(d)["card"],
                 next(c["card"] for c in fake_bot.reply_chain.stream(d) if "card" in c)]:
        assert set(card) == CARD_KEYS and (card["team"], card["priority"]) == ("food_safety", "P1")
        assert card["summary_en"] and card["next_step"]

def test_check_card_unit(fake_bot):
    x = {"reasons": ["amount>1000"], "raw": {"team": "support", "priority": "P3", "summary_en": "", "next_step": None}}
    card = fake_bot.check_card(x)
    assert card["summary_en"] == "Card not generated by AI; read the chat." and card["next_step"]
    assert fake_bot.check_card({"reasons": ["food safety"], "raw": None})["team"] == "food_safety"

def test_card_llm_down_gives_default_card(fake_bot):
    FAKE["script"]["card"] = boom(auth_error)
    card = card_of(fake_bot, BY_ID["T13"]["message"])
    assert card["team"] == "finance" and card["summary_en"].startswith("Card not generated")

def test_t01_no_card_and_no_extra_llm_call(fake_bot):
    out = fake_bot.reply_chain.invoke(decide(fake_bot, BY_ID["T01"]["message"]))
    assert out == {"reply": "Fake reply.", "card": None}
    assert calls("card") == [] and [c["kind"] for c in calls()].count("lane") == 1
    assert len(calls()) == 3  # intent + language + lane, nothing else

def test_only_human_review_gets_cards(fake_bot):
    for t in TICKETS:
        card = card_of(fake_bot, t["message"])
        if t["expected_route"] == "human_review":
            assert set(card) == CARD_KEYS and card["team"] in fake_bot.TEAMS and card["priority"] in fake_bot.PRIORITIES
        else:
            assert card is None
    assert len(calls("card")) == sum(t["expected_route"] == "human_review" for t in TICKETS)

def test_card_prompt_sees_masked_text_note_and_history(fake_bot):
    hist = [HumanMessage(content="FK1009 ki biryani kharab thi, call 9845012345"), AIMessage(content="Sorry to hear.")]
    card_of(fake_bot, "ab bacche ko ulti ho rahi hai", hist)
    c = calls("card")[-1]
    text = " ".join(m.content for m in c["messages"])
    assert "biryani" in text and "[PHONE]" in text and "9845012345" not in text
    assert "order=FK1009" in c["system"] and "reasons=['food safety']" in c["system"]

def test_stream_yields_reply_text_then_one_card(fake_bot):
    chunks = list(fake_bot.reply_chain.stream(decide(fake_bot, BY_ID["T09"]["message"])))
    cards = [c["card"] for c in chunks if "card" in c]
    assert "".join(c.get("reply", "") for c in chunks) == "Fake reply."
    assert len(cards) == 1 and cards[0]["team"] == "food_safety"
    none_chunks = [c for c in fake_bot.reply_chain.stream(decide(fake_bot, BY_ID["T01"]["message"])) if "card" in c]
    assert none_chunks == [{"card": None}]

def test_reply_and_card_run_in_parallel(fake_bot):
    d = decide(fake_bot, BY_ID["T09"]["message"])
    FAKE["delay"].update(lane=0.4, card=0.4)
    t0 = time.perf_counter()
    fake_bot.reply_chain.invoke(d)
    assert time.perf_counter() - t0 < 0.7

def test_memory_stores_reply_text_not_dict(fake_bot):
    cfg = {"configurable": {"session_id": "ex4"}}
    out = fake_bot.bot.invoke({"input": BY_ID["T09"]["message"]}, config=cfg)
    assert out["reply"] == "Fake reply." and out["card"]["team"] == "food_safety"
    assert [m.content for m in fake_bot.get_history("ex4").messages] == [BY_ID["T09"]["message"], "Fake reply."]
    chunks = list(fake_bot.bot.stream({"input": "aur kab tak?"}, config=cfg))
    assert "".join(c.get("reply", "") for c in chunks) == "Fake reply." and len(fake_bot.get_history("ex4").messages) == 4

def test_ui_shows_card_and_ops_columns(fake_bot):
    at = app()
    at.chat_input[0].set_value(BY_ID["T09"]["message"]).run()
    assert not at.exception
    assert at.chat_message[1].markdown[0].value == "Fake reply."  # text, not a dict
    assert any("Hand-off card" in md.value for md in at.expander[0].markdown)
    assert at.session_state["messages"][1]["card"]["team"] == "food_safety"
    queue = at.dataframe[0].value
    assert {"team", "priority", "summary"} <= set(queue.columns) and queue["team"].tolist() == ["food_safety"]

def test_ui_no_card_for_auto_refund(fake_bot):
    at = app()
    at.chat_input[0].set_value(BY_ID["T01"]["message"]).run()
    assert not any("Hand-off card" in md.value for md in at.expander[0].markdown)
    assert at.session_state["messages"][1]["card"] is None

ASCII_SENTENCE = re.compile(r"^[\x00-\x7F₹]+$")

@live
@pytest.mark.parametrize("tid,team", [("T09", "food_safety"), ("T13", "finance"), ("T10", "rider_ops")])
def test_live_cards(real_bot, tid, team):
    card = card_of(real_bot, BY_ID[tid]["message"])
    assert card["team"] == team and ASCII_SENTENCE.match(card["summary_en"])

@live
def test_live_x4_card_summary_in_english(real_bot):
    card = card_of(real_bot, X["X4"])
    assert (card["team"], card["priority"]) == ("food_safety", "P1") and ASCII_SENTENCE.match(card["summary_en"])

# ---- Ex5: bulk triage ----
ROW_KEYS = {"id", "order_id", "intent", "language", "route", "reasons"}

def test_bulk_triage_18_sample_tickets(fake_bot):
    rows = fake_bot.bulk_triage.invoke(TICKETS)
    assert [r["id"] for r in rows] == [t["id"] for t in TICKETS]
    assert all(set(r) == ROW_KEYS for r in rows)
    assert [r["route"] for r in rows] == [t["expected_route"] for t in TICKETS]
    assert len(calls("lane")) == 0  # routed only, no replies

def test_bulk_triage_reads_csv_file(fake_bot):
    import pandas as pd
    rows = pd.read_csv(ROOT / "data" / "tickets.csv", dtype=str, keep_default_na=False).to_dict("records")
    assert [r["route"] for r in fake_bot.bulk_triage.invoke(rows)] == [t["expected_route"] for t in TICKETS]

@pytest.mark.parametrize("bad", [{"id": "E1", "message": ""}, {"id": "E2", "message": "   \n"}, {"id": "E3"},
                                 {"id": "E4", "message": None}])
def test_empty_row_is_error_and_others_fine(fake_bot, bad):
    rows = fake_bot.bulk_triage.invoke(TICKETS[:9] + [bad] + TICKETS[9:])
    broken = rows[9]
    assert broken["id"] == bad["id"] and broken["route"] == "error" and "empty message" in broken["reasons"]
    good = rows[:9] + rows[10:]
    assert [r["route"] for r in good] == [t["expected_route"] for t in TICKETS]
    assert len(calls("intent")) == 18  # no LLM call for the empty row

def test_llm_failure_on_one_row_does_not_stop_others(fake_bot):
    def flaky_intent(call):
        if "POISON" in call["human"]: raise auth_error()
        return fake_intent(call)
    FAKE["script"]["intent"] = flaky_intent
    rows = fake_bot.bulk_triage.invoke([{"id": "P1", "message": "POISON FK1001 bread missing"}] + TICKETS)
    assert rows[0]["route"] == "human_review" and rows[0]["intent"] == "unclear"  # Ex3: unclear, not a crash
    assert [r["route"] for r in rows[1:]] == [t["expected_route"] for t in TICKETS]

def test_to_rows_turns_exceptions_into_error_rows(fake_bot):
    ok = decide(fake_bot, BY_ID["T01"]["message"])
    out = fake_bot.to_rows({"rows": [{"id": "A"}, {"id": "B"}], "results": [ok, KeyError("order_id")]})
    assert out[0] == {"id": "A", "order_id": "FK1001", "intent": "missing_item", "language": "hinglish",
                      "route": "auto_refund", "reasons": "amount<=300, clean customer"}
    assert out[1]["route"] == "error" and out[1]["reasons"] == "KeyError: 'order_id'" and out[1]["order_id"] is None

def test_require_message(fake_bot):
    assert fake_bot.require_message({"input": "hi"}) == {"input": "hi"}
    with pytest.raises(ValueError, match="empty message"): fake_bot.require_message({"input": "  "})

def test_bulk_batch_is_at_least_2x_faster_than_a_loop(fake_bot):
    FAKE["delay"].update(intent=0.05, language=0.05)
    t0 = time.perf_counter()
    for t in TICKETS: fake_bot.router.invoke({"input": t["message"], "chat_history": []})
    loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    fake_bot.bulk_triage.invoke(TICKETS)
    batch = time.perf_counter() - t0
    assert batch * 2 <= loop, (batch, loop)

def test_max_concurrency_from_config_is_respected(fake_bot):
    import threading
    active, peak, lock = [0], [0], threading.Lock()
    def counting(call):
        with lock: active[0] += 1; peak[0] = max(peak[0], active[0])
        time.sleep(0.03)
        with lock: active[0] -= 1
        return fake_intent(call)
    FAKE["script"]["intent"] = counting
    fake_bot.bulk_triage.invoke(TICKETS)
    assert 1 < peak[0] <= 5
    peak[0] = 0
    fake_bot.bulk_triage.invoke(TICKETS, config={"max_concurrency": 1})
    assert peak[0] == 1

def test_bulk_triage_handles_a_big_backlog(fake_bot):
    backlog = [{"id": f"{t['id']}-{i}", "message": t["message"]} for i in range(6) for t in TICKETS]  # 108 tickets
    rows = fake_bot.bulk_triage.invoke(backlog)
    assert len(rows) == 108 and [r["route"] for r in rows] == [t["expected_route"] for t in TICKETS] * 6

def test_ui_bulk_triage_sample_button(fake_bot):
    at = app()
    next(b for b in at.button if b.label == "Triage the 18 sample tickets").click().run()
    assert not at.exception
    result = at.session_state["bulk"]["result"]
    assert len(result) == 18 and (result.route == [t["expected_route"] for t in TICKETS]).all()
    assert any("Matches expected_route: 18/18" in c.value for c in at.caption)
    assert any("18 tickets routed in" in c.value for c in at.caption)

def test_ui_has_csv_uploader(fake_bot):
    at = app()
    assert at.get("file_uploader")

@live
def test_live_bulk_triage_18_and_speed(real_bot):
    t0 = time.perf_counter()
    for t in TICKETS: real_bot.router.invoke({"input": t["message"], "chat_history": []})
    loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = real_bot.bulk_triage.invoke(TICKETS + [{"id": "X5", "message": ""}])
    batch = time.perf_counter() - t0
    print(f"\nfor-loop {loop:.1f}s vs batch {batch:.1f}s")
    assert [r["route"] for r in rows[:18]] == [t["expected_route"] for t in TICKETS] and rows[18]["route"] == "error"
    assert batch * 2 <= loop

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
