"""FreshKart support bot: the notebook (freshkart_lcel.ipynb) moved into a .py file."""
import os, re, sys, threading, time
import openai
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch, ConfigurableField
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Cell 2: Setup (data paths are relative to this file)
load_dotenv()
DATA = Path(__file__).parent / "data"

def get_llm(model=None):
    # Ex 3: max_retries=0 so our own .with_retry is the one that retries; timeout so a hung call fails (and falls back).
    return ChatOpenAI(model=model or os.getenv("MODEL", "openai/gpt-4o-mini"),
                      base_url="https://openrouter.ai/api/v1",
                      api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0, max_retries=0, timeout=20)

def read(name):
    return pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False)

# Ex 6: model switcher at runtime (configurable_alternatives). ONE llm object whose model is picked per request with
# config={"configurable": {"model": "llama"}}. Every chain built on llm (intent, language, lanes, card, Ex3's retry and
# fallback wrappers) gets the switch for free: config travels down to every child Runnable on each call, and only the
# alternative chosen by the config is called. No chain is rebuilt.
MODELS = {"mini": os.getenv("MODEL", "openai/gpt-4o-mini"), "llama": "meta-llama/llama-3.1-8b-instruct",
          "strong": "openai/gpt-4o"}
llm = get_llm(MODELS["mini"]).configurable_alternatives(
    ConfigurableField(id="model", name="Model"), default_key="mini",
    **{key: get_llm(name) for key, name in MODELS.items() if key != "mini"})

# Ex 3: never-down LLM (.with_retry + .with_fallbacks). Retry only errors that can succeed on a second try (timeout,
# connection, 429, 5xx); a wrong model name (404) or a bad key (401) goes straight to the backup model.
# Note: RunnableRetry only wraps invoke/batch. On .stream() (the lanes in the app) there is no retry, only fallbacks.
BACKUP_MODEL = os.getenv("BACKUP_MODEL", "meta-llama/llama-3.1-8b-instruct")
RETRYABLE = (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
RETRY_WAIT = float(os.getenv("RETRY_WAIT", "0.5"))  # seconds before the first retry (tests set 0)

def retrying(model):
    return model.with_retry(retry_if_exception_type=RETRYABLE, stop_after_attempt=3,
                            exponential_jitter_params={"initial": RETRY_WAIT, "jitter": RETRY_WAIT})

safe_llm = retrying(llm).with_fallbacks([retrying(get_llm(BACKUP_MODEL))])
orders, customers, inventory, riders, tickets = [read(n) for n in
                                                 ["orders", "customers", "inventory", "riders", "tickets"]]
TICKETS = tickets.to_dict("records")

# Cell 10: RunnableLambda (order id by regex)
def find_order_id(text):
    m = re.search(r"(?i)\b(?:FK|ord_)\s?(\d{4})\b", text)
    return "FK" + m.group(1) if m else None

# Cell 12: data lookups and regex signals
def row(df, col, val):
    m = df[df[col] == val]
    return m.iloc[0].to_dict() if len(m) else None

def get_order(order_id): return row(orders, "order_id", order_id)

def get_customer(order_id):
    o = get_order(order_id)
    return row(customers, "customer_id", o["customer_id"]) if o else None

def check_inventory(order_id):
    o = get_order(order_id)
    if not o: return None
    d = {"expired_confirmed": False, "baby": False}
    for item in o["items"].split(";"):
        r = row(inventory[inventory.dark_store == o["dark_store"]], "name", item)
        if r:
            d["expired_confirmed"] |= r["expiry_date"] < o["delivered_at"][:10]
            d["baby"] |= r["category"] == "baby_care"
    return d

def get_rider(order_id): return row(riders, "order_id", order_id)

SAFETY = (r"(?i)\b(?:food poisoning|vomit\w*|ulti|diarrhoea|diarrhea|allerg\w*|rash|illness|sick|"
          r"stomach pain|fever|hospital|infection|swelling|choking|fungus|mould|mold)\b")
def has_safety(text): return bool(re.search(SAFETY, text))

def claimed_amount(text):
    nums = re.findall(r"(?i)(?:₹|Rs\.?|INR)\s?(\d[\d,]*(?:\.\d+)?)", text)
    return max(float(n.replace(",", "")) for n in nums) if nums else None

# Cell 14: RunnablePassthrough.assign + memory (look back through chat_history)
def order_id_from(x):
    for text in [x["input"]] + [m.content for m in reversed(x.get("chat_history", []))]:
        if find_order_id(text): return find_order_id(text)

# Ex 1: PII shield (RunnableLambda + RunnablePassthrough.assign). The first station on the belt adds masked_input
# (and masks chat_history) so prompts never see phone, email, UPI or card numbers. The regex signals keep reading
# the original input, so routing does not change. Order matters: email before UPI (both have @), and phone before
# card (a +91 phone has 12 digits, so the card pattern would eat it). FK1001 and Rs 520 are too short to match.
PII = [
    (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[EMAIL]"),
    (r"[\w.-]+@[a-zA-Z]+\b", "[UPI]"),
    (r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\s-]?\d)", "[PHONE]"),
    (r"(?<!\d)(?:\d[ -]?){11,15}\d(?!\d)", "[CARD]"),
]

def mask_pii(text):
    count = 0
    for pattern, tag in PII:
        text, n = re.subn(pattern, tag, text)
        count += n
    return text, count

def mask_history(x):
    return [m.model_copy(update={"content": mask_pii(m.content)[0]}) for m in x.get("chat_history", [])]

mask = RunnablePassthrough.assign(
    masked_input=RunnableLambda(lambda x: mask_pii(x["input"])[0]),
    pii_count=RunnableLambda(lambda x: mask_pii(x["input"])[1]),
    chat_history=RunnableLambda(mask_history),
)
masked = RunnableLambda(lambda x: {"input": x["masked_input"]})  # adapter: prompts read the masked text as {input}

# Ex 7: live pipeline tracker. Every station gets a readable name (.with_config(run_name=...)) and listeners
# (.with_listeners(on_start, on_end)) that append (name, start, end, thread) to a list the CALLER passes in
# config={"configurable": {"trace": [...]}}. No trace list in the config = nothing recorded: the tracker is optional
# and not part of the logic.
# Why listeners and not astream_events: listeners get the run's real start_time/end_time. With astream_events every
# step of a sequence "starts" at t=0 (it is waiting for its input), so decide_route would look as slow as the LLM.
# Listeners also fire on .stream() (the lanes) and never surface a lane's draft tokens (the Bonus guard needs that).
# Watch out: listeners of the RunnableParallel stations run on worker threads, so they only append to the list;
# app.py polls that list from Streamlit's own thread and draws it (no st.* call in a listener).
SIGNAL_STATIONS = ["intent (LLM)", "language (LLM)", "order lookup", "customer lookup", "inventory", "rider",
                   "safety keywords", "claimed amount"]

def on_start(run, config):
    trace = config.get("configurable", {}).get("trace")
    if trace is not None:
        trace.append({"id": run.id, "name": run.name, "start": run.start_time, "end": None,
                      "thread": threading.current_thread().name})

def on_end(run, config):
    trace = config.get("configurable", {}).get("trace")
    for entry in trace if trace is not None else []:
        if entry["id"] == run.id: entry["end"] = run.end_time

def station(name, runnable):
    return runnable.with_config(run_name=name).with_listeners(on_start=on_start, on_end=on_end)

def trace_table(trace):
    """Rows for display: seconds since the first station started, and how long each station took (None = running)."""
    if not trace: return []
    t0 = min(e["start"] for e in trace)
    rows = [{"station": e["name"], "at_s": round((e["start"] - t0).total_seconds(), 2),
             "seconds": round((e["end"] - e["start"]).total_seconds(), 2) if e["end"] else None,
             "thread": e["thread"]} for e in list(trace)]
    return sorted(rows, key=lambda r: (r["seconds"] is None, r["at_s"] + (r["seconds"] or 0)))

# Cell 16: RunnableParallel (all signals at once; only intent calls the LLM)
INTENTS = """wismo: where is my order, delivery status, delay
missing_item: an item is not in the bag
damaged_spoiled: damaged, broken, leaking or spoiled item
expired: item is past its expiry date
wrong_item_substitution: wrong item or unwanted substitution / different brand
quality_complaint: poor quality (limp veg, stale) but not clearly spoiled
refund_status: where is my refund
payment_issue: double debit, payment failed but money cut
coupon_pass: coupon, FreshKart Pass, offers
food_safety: illness, allergy, contamination, foreign object
rider_misconduct: rude rider, demanded cash, harassment
general: anything else, FAQs"""

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the customer message into exactly one intent:\n" + INTENTS +
     "\nAlso give the sentiment: angry, neutral or happy.\nAnswer ONLY as: intent,sentiment"),
    ("human", "{input}"),
])

def parse_intent(text):
    label = text.strip().lower().split(",")[0].strip()
    return label if label in [l.split(":")[0] for l in INTENTS.split("\n")] else "unclear"

# Ex 2: language signal. A second LLM chain added as a new key of the same RunnableParallel, so it runs at the same
# time as intent (prompt | llm | StrOutputParser | RunnableLambda(clean_up)). The clean-up falls back to "english".
LANGS = ["english", "hinglish", "hindi", "kannada", "telugu", "other"]
language_prompt = ChatPromptTemplate.from_messages([
    ("system", "Which language is the customer message written in? Choose exactly one: " + ", ".join(LANGS) + ".\n"
               "hinglish = Hindi words written in English letters, often mixed with English. hindi = Devanagari script.\n"
               "Answer ONLY with the label."),
    ("human", "{input}"),
])

def parse_language(text):
    words = re.findall(r"[a-z]+", text.lower())
    return words[0] if words and words[0] in LANGS else "english"

def ai_down(text):
    return AIMessage(content=text, response_metadata={"model_name": "none (AI down)"})

def unpack_intent(x):
    s = dict(x["signals"])
    msg = s.pop("intent_msg")
    s.update(intent=parse_intent(msg.content), intent_model=msg.response_metadata.get("model_name", "unknown"))
    return {**x, "signals": s}

oid = lambda f: RunnableLambda(lambda x: f(x["order_id"]))
signals = RunnableParallel(
    # Ex 3: intent keeps the AIMessage (to record which model answered); unpack_intent parses it after the parallel step.
    # Last fallback: "unclear", which decide_route already sends to human_review.
    intent_msg=station("intent (LLM)", (masked | intent_prompt | safe_llm)
                       .with_fallbacks([RunnableLambda(lambda x: ai_down("unclear"))])),
    language=station("language (LLM)", (masked | language_prompt | safe_llm | StrOutputParser()
                                        | RunnableLambda(parse_language)).with_fallbacks([RunnableLambda(lambda x: "english")])),
    order=station("order lookup", oid(get_order)), customer=station("customer lookup", oid(get_customer)),
    inventory=station("inventory", oid(check_inventory)), rider=station("rider", oid(get_rider)),
    safety=station("safety keywords", RunnableLambda(lambda x: has_safety(x["input"]))),
    claimed_amount=station("claimed amount", RunnableLambda(lambda x: claimed_amount(x["input"]))),
)

# Cell 18: decide_route (rules in plain Python, first match wins)
NOW = pd.Timestamp("2026-10-12 20:00")
CLAIMS = {"missing_item", "damaged_spoiled", "expired", "wrong_item_substitution", "quality_complaint"}
INFO = {"refund_status", "coupon_pass", "general"}
LABELS = CLAIMS | INFO | {"wismo", "payment_issue", "food_safety", "rider_misconduct"}

def decide_route(x):
    s = x["signals"]
    i, o, c, r, inv = s["intent"], s["order"], s["customer"], s["rider"], s["inventory"]
    amount = s["claimed_amount"] if s["claimed_amount"] is not None else (float(o["amount"]) if o and i in CLAIMS else 0)
    def go(route, why): return {**x, "route": route, "reasons": [why]}

    if s["safety"] or i == "food_safety": return go("human_review", "food safety")
    if i == "rider_misconduct": return go("human_review", "rider misconduct")
    if i == "payment_issue": return go("human_review", "payment issue (Finance)")
    if amount > 1000: return go("human_review", "amount>1000")
    if i not in LABELS: return go("human_review", "unclear intent")
    if i in INFO: return go("info", "information request")
    if o is None: return go("info", "no order found, ask for order id")
    if o["status"] == "delivered" and r and r["gps_delivered"] == "n": return go("human_review", "OMS delivered but GPS says no")
    if i == "wismo": return go("wismo", "delivery status question")
    if float(c["fraud_score"]) > 0.7 or int(c["refunds_30d"]) > 3 or inv["baby"]:
        return go("human_review", "fraud score, refunds_30d or baby product")
    if o["status"] != "delivered": return go("policy_decline", "order not delivered")
    if (NOW - pd.Timestamp(o["delivered_at"])) > pd.Timedelta(hours=48): return go("policy_decline", "outside 48 hours")
    if i == "expired" and not inv["expired_confirmed"]: return go("policy_decline", "expiry not confirmed")
    if i == "wrong_item_substitution" and o["substituted"] == "y": return go("substitution", "substituted order")
    if amount <= 300: return go("auto_refund", "amount<=300, clean customer")
    return go("human_review", "amount between 301 and 1000")

# Cell 20: RunnableBranch (one reply style per lane)
def add_note(x):
    s, o = x["signals"], x["signals"]["order"]
    facts = [f"route={x['route']}", f"reasons={x['reasons']}", f"intent={s['intent']}"]
    if o:
        delay = int(o["actual_min"]) - int(o["promised_min"]) if o["actual_min"] else None
        facts += [f"order={o['order_id']}", f"status={o['status']}", f"amount=Rs{o['amount']}", f"payment={o['payment_mode']}",
                  f"items={o['items']}", f"promised_min={o['promised_min']}", f"actual_min={o['actual_min'] or 'n/a'}",
                  f"substituted={o['substituted']}"]
        if o["status"] == "delivered" and delay is not None:
            facts.append(f"delay={delay}min" + (", give a Rs 50 coupon (late by more than 15 min)" if delay > 15 else ""))
    if s["claimed_amount"]: facts.append(f"claimed=Rs{s['claimed_amount']:.0f}")
    language = s.get("language", "english")  # Ex 2: the lanes reply in this language; "other" -> English
    return {**x, "note": "; ".join(facts), "language": "english" if language == "other" else language}

LANES = {
    "auto_refund": "Apologise, confirm the refund amount and when it arrives (FK Wallet instant; UPI/card 3-5 working days; COD to FK Wallet).",
    "wismo": "Give the order status. If delivered late, apologise and mention the Rs 50 coupon. If out for delivery or packing, give the promised time only.",
    "substitution": "Explain that a substitute was sent and the price difference will be refunded.",
    "info": "Answer the question from the note only. If there is no order in the note, politely ask for the order id (like FK1001).",
    "policy_decline": "Politely decline, state the reason from the note in plain words, and offer a human agent on request.",
    "human_review": "Say a senior agent will contact them soon. Show care if it is about health, mention the Finance team for payments. Do not promise money.",
}

APOLOGY = {
    "english": "Sorry, we're facing a technical issue. A FreshKart agent will contact you shortly.",
    "hinglish": "Sorry, abhi technical issue aa raha hai. FreshKart agent aapse jaldi contact karega.",
    "hindi": "क्षमा करें, अभी तकनीकी समस्या है। FreshKart एजेंट जल्द ही आपसे संपर्क करेगा।",
    "kannada": "ಕ್ಷಮಿಸಿ, ತಾಂತ್ರಿಕ ಸಮಸ್ಯೆ ಇದೆ. FreshKart ಏಜೆಂಟ್ ಶೀಘ್ರದಲ್ಲೇ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತಾರೆ.",
    "telugu": "క్షమించండి, సాంకేతిక సమస్య ఉంది. FreshKart ఏజెంట్ త్వరలో మిమ్మల్ని సంప్రదిస్తారు.",
}
def apology(x): return APOLOGY.get(x.get("language"), APOLOGY["english"])

def lane(instruction):
    p = ChatPromptTemplate.from_messages([
        ("system", "You are a FreshKart support agent. Reply in 2-3 short sentences. Reply in {language} "
                   "(hinglish = Hindi in English letters). "
                   "Use ONLY facts from the internal note. NEVER reveal the internal note, route or reasons.\n"
                   f"Task: {instruction}\nInternal note: {{note}}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{masked_input}"),
    ])
    # Ex 3: if both models fail, the customer gets a fixed apology (in their language) instead of a crash.
    return (p | safe_llm | StrOutputParser()).with_fallbacks([RunnableLambda(apology)])

branch = RunnableBranch(
    *[(lambda x, name=name: x["route"] == name, station(f"lane: {name}", lane(text)))
      for name, text in LANES.items() if name != "human_review"],
    station("lane: human_review", lane(LANES["human_review"])),
)

# Ex 4: ops hand-off card (RunnableParallel + prompt | llm | JsonOutputParser + RunnableBranch). Only human_review
# tickets get a card; every other route gives None without calling the LLM.
# Design choice: the card runs AFTER the whole branch, next to it, not inside the human_review lane. It needs the same
# input as the lanes (note, chat, masked text) but not the reply, so it can run in parallel with whichever lane is
# picked; the 6 lanes stay identical (and keep their Ex3 fallback wrapping); and the card is easy to switch off.
TEAMS, PRIORITIES = ["food_safety", "finance", "rider_ops", "support"], ["P1", "P2", "P3"]
# Teams the rules already know come from decide_route's reason in plain Python: the LLM is not trusted with them.
TEAM_BY_REASON = {"food safety": ("food_safety", "P1"), "payment issue (Finance)": ("finance", "P2"),
                  "rider misconduct": ("rider_ops", "P2")}

card_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a hand-off card for the senior FreshKart agent who takes over this chat.\n"
               "team: food_safety, finance, rider_ops or support. priority: P1 (health or safety risk), "
               "P2 (money or rider issue), P3 (anything else).\n"
               "Answer ONLY with JSON with keys team, priority, summary_en, next_step. summary_en is one sentence in "
               "English, even if the customer wrote in another language. next_step is one short instruction.\n"
               "Internal note: {note}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{masked_input}"),
])

def default_card(x):
    team, priority = TEAM_BY_REASON.get(x["reasons"][0], ("support", "P2"))
    return {"team": team, "priority": priority, "summary_en": "Card not generated by AI; read the chat.",
            "next_step": "Read the chat and contact the customer."}

def check_card(x):
    card = x.get("raw")
    if not isinstance(card, dict): return default_card(x)
    team = card.get("team") if card.get("team") in TEAMS else "support"
    priority = card.get("priority") if card.get("priority") in PRIORITIES else "P2"
    team, priority = TEAM_BY_REASON.get(x["reasons"][0], (team, priority))
    return {"team": team, "priority": priority,
            "summary_en": str(card.get("summary_en") or default_card(x)["summary_en"]),
            "next_step": str(card.get("next_step") or default_card(x)["next_step"])}

card_chain = (RunnablePassthrough.assign(raw=card_prompt | safe_llm | JsonOutputParser())
              | RunnableLambda(check_card)).with_fallbacks([RunnableLambda(default_card)])
card_branch = RunnableBranch((lambda x: x["route"] == "human_review", station("hand-off card", card_chain)),
                             RunnableLambda(lambda x: None))

# Cell 22: full chain + memory. The app uses router (decide) then reply_chain (stream) to avoid a 2nd intent call.
router = station("router", station("pii mask", mask)
                 | RunnablePassthrough.assign(order_id=station("order id", RunnableLambda(order_id_from)))
                 | RunnablePassthrough.assign(signals=station("signals (parallel)", signals))
                 | station("unpack_intent", RunnableLambda(unpack_intent)) | station("decide_route", RunnableLambda(decide_route)))
reply_chain = RunnableLambda(add_note) | RunnableParallel(reply=branch, card=card_branch)  # Ex 4: {"reply", "card"}
full_chain = router | reply_chain

# Ex 5: bulk triage for the morning backlog. router.batch(max_concurrency, return_exceptions=True) routes every ticket
# (no replies); a RunnableLambda turns the results, some of which may be exceptions, into plain row dicts.
# Since Ex3 an LLM outage no longer raises (the ticket becomes "unclear" -> human_review), so "error" rows come from
# input problems. require_message makes an empty message one of those: it fails fast with no LLM call.
def require_message(x):
    if not x["input"].strip(): raise ValueError("empty message")
    return x

bulk_row = RunnableLambda(require_message) | router

def run_batch(x, config):
    inputs = [{"input": str(r.get("message") or ""), "chat_history": []} for r in x["rows"]]
    return bulk_row.batch(inputs, config={**config, "max_concurrency": config.get("max_concurrency") or 5},
                          return_exceptions=True)

def to_rows(x):
    def one(row, res):
        if isinstance(res, Exception):
            return {"id": row.get("id"), "order_id": None, "intent": None, "language": None, "route": "error",
                    "reasons": f"{type(res).__name__}: {res}"}
        return {"id": row.get("id"), "order_id": res["order_id"], "intent": res["signals"]["intent"],
                "language": res["signals"]["language"], "route": res["route"], "reasons": "; ".join(res["reasons"])}
    return [one(row, res) for row, res in zip(x["rows"], x["results"])]

bulk_triage = (RunnableLambda(lambda rows: {"rows": rows}) | RunnablePassthrough.assign(results=RunnableLambda(run_batch))
               | RunnableLambda(to_rows))

# Ex 6: compare models on the 18 sample tickets with the SAME router object; only the config changes.
# Tokens come from UsageMetadataCallbackHandler (routing calls only: intent + language). Prices are approximate
# OpenRouter list prices in USD per 1M tokens (input, output); edit them if they change.
PRICE_USD_PER_M = {"openai/gpt-4o-mini": (0.15, 0.60), "meta-llama/llama-3.1-8b-instruct": (0.02, 0.03),
                   "openai/gpt-4o": (2.50, 10.00)}
USD_INR = 88

def rupees(usage):
    total = 0.0
    for name, u in usage.items():
        price = next((PRICE_USD_PER_M[k] for k in sorted(PRICE_USD_PER_M, key=len, reverse=True) if name.startswith(k)), (0, 0))
        total += (u.get("input_tokens", 0) * price[0] + u.get("output_tokens", 0) * price[1]) / 1e6 * USD_INR
    return total

def evaluate_model(key):
    usage = UsageMetadataCallbackHandler()
    t0 = time.perf_counter()
    results = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS],
                           config={"configurable": {"model": key}, "max_concurrency": 5, "callbacks": [usage]},
                           return_exceptions=True)
    ok = [not isinstance(r, Exception) and r["route"] == t["expected_route"] for r, t in zip(results, TICKETS)]
    return {"model": key, "model_name": MODELS[key], "score": sum(ok), "of": len(TICKETS),
            "seconds": round(time.perf_counter() - t0, 1),
            "tokens": sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values()),
            "rs_per_ticket": round(rupees(usage.usage_metadata) / len(TICKETS), 4),
            "answered_by": ", ".join(sorted({r["signals"]["intent_model"] for r in results if not isinstance(r, Exception)}))}

evaluate = RunnableLambda(evaluate_model)

store = {}
def get_history(session_id):
    return store.setdefault(session_id, InMemoryChatMessageHistory())

bot = RunnableWithMessageHistory(full_chain, get_history, input_messages_key="input", history_messages_key="chat_history",
                                 output_messages_key="reply")

if __name__ == "__main__":  # python bot.py T09
    t = next(t for t in TICKETS if t["id"] == (sys.argv[1] if len(sys.argv) > 1 else "T01"))
    print("Customer:", t["message"], "\nBot: ", end="")
    for chunk in bot.stream({"input": t["message"]}, config={"configurable": {"session_id": "cli"}}):
        print(chunk.get("reply", ""), end="", flush=True)
        if chunk.get("card"): print("\nHand-off card:", chunk["card"])
    print()
