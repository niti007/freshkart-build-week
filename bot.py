"""FreshKart support bot: the notebook (freshkart_lcel.ipynb) moved into a .py file."""
import os, re, sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch, ConfigurableField
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Cell 2: Setup (data paths are relative to this file)
load_dotenv()
DATA = Path(__file__).parent / "data"

mini_llm = ChatOpenAI(model=os.getenv("MODEL", "openai/gpt-4o-mini"),
                      base_url="https://openrouter.ai/api/v1",
                      api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0, max_retries=0)
llama_llm = ChatOpenAI(model=os.getenv("BACKUP_MODEL", "meta-llama/llama-3.1-8b-instruct"),
                       base_url="https://openrouter.ai/api/v1",
                       api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0, max_retries=0)

llm = mini_llm.configurable_alternatives(
    ConfigurableField(id="model"),
    default_key="mini",
    llama=llama_llm,
)
backup_llm = llama_llm
def read(name):
    return pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False)

orders, customers, inventory, riders, tickets = [read(n) for n in
                                                 ["orders", "customers", "inventory", "riders", "tickets"]]
TICKETS = tickets.to_dict("records")

# Cell 9: PII Masking (Exercise 1)
def mask_pii(text):
    count = 0
    def sub_card(m): nonlocal count; count += 1; return "[CARD]"
    text = re.sub(r"\b(?:\d[ -]?){11,15}\d\b", sub_card, text)
    def sub_email(m): nonlocal count; count += 1; return "[EMAIL]"
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", sub_email, text)
    def sub_upi(m): nonlocal count; count += 1; return "[UPI]"
    text = re.sub(r"\b[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}\b", sub_upi, text)
    def sub_phone(m): nonlocal count; count += 1; return "[PHONE]"
    text = re.sub(r"\b(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}\b", sub_phone, text)
    return text, count

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

LANGUAGES = {"english", "hinglish", "hindi", "kannada", "telugu", "other"}
VALID_INTENTS = [l.split(":")[0] for l in INTENTS.split("\n")]

nlu_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify customer text into:\n"
               "1. Intent:\n" + INTENTS + "\n"
               "2. Sentiment: angry, neutral, happy\n"
               "3. Language: english, hinglish (Hindi words written using Latin/English alphabet, or code-mixed Hindi-English), hindi (ONLY if written in Devanagari script like हिन्दी), kannada, telugu, other\n"
               "Answer ONLY as: intent,sentiment,language"),
    ("human", "{input}"),
])

def parse_nlu(text, model_name="main"):
    parts = [p.strip().lower() for p in text.strip().split(",")]
    intent = parts[0] if len(parts) > 0 and parts[0] in VALID_INTENTS else "unclear"
    sentiment = parts[1] if len(parts) > 1 and parts[1] in {"angry", "neutral", "happy"} else "neutral"
    language = parts[2] if len(parts) > 2 and parts[2] in LANGUAGES else "english"
    return {"intent": intent, "sentiment": sentiment, "language": language, "model_used": model_name}

primary_nlu = (nlu_prompt | llm | StrOutputParser() | RunnableLambda(lambda t: parse_nlu(t, "main"))).with_retry(stop_after_attempt=2)
backup_nlu = (nlu_prompt | backup_llm | StrOutputParser() | RunnableLambda(lambda t: parse_nlu(t, "backup"))).with_retry(stop_after_attempt=2)
offline_nlu = RunnableLambda(lambda _: {"intent": "unclear", "sentiment": "neutral", "language": "english", "model_used": "offline_fallback"})

nlu_chain = RunnableLambda(lambda x: {"input": x["masked_input"]}) | primary_nlu.with_fallbacks([backup_nlu, offline_nlu])

# Cell 19: Live Pipeline Tracker (Exercise 7)
def make_tracker_listener():
    def on_end(run, config):
        tracker = config.get("configurable", {}).get("tracker")
        if tracker is not None:
            dur = (run.end_time - run.start_time).total_seconds() if (run.end_time and run.start_time) else 0.0
            tracker.append({
                "name": run.name,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration": dur,
                "outputs": getattr(run, "outputs", None),
            })
    return on_end

tracker_listener = make_tracker_listener()

def track(r, name):
    return r.with_config(run_name=name).with_listeners(on_end=tracker_listener)

oid = lambda f: RunnableLambda(lambda x: f(x["order_id"]))
signals = RunnableParallel(
    intent=RunnableLambda(lambda x: x["nlu"]["intent"]).with_config(run_name="signal_intent"),
    language=RunnableLambda(lambda x: x["nlu"]["language"]).with_config(run_name="signal_language"),
    sentiment=RunnableLambda(lambda x: x["nlu"]["sentiment"]).with_config(run_name="signal_sentiment"),
    model_used=RunnableLambda(lambda x: x["nlu"]["model_used"]).with_config(run_name="signal_model_used"),
    order=track(oid(get_order), "order_lookup"),
    customer=track(oid(get_customer), "customer_lookup"),
    inventory=track(oid(check_inventory), "inventory_lookup"),
    rider=track(oid(get_rider), "rider_lookup"),
    safety=track(RunnableLambda(lambda x: has_safety(x["input"])), "safety_check"),
    claimed_amount=track(RunnableLambda(lambda x: claimed_amount(x["input"])), "amount_extract"),
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
    lang = s.get("language", "english")
    target_lang = "english" if lang == "other" else lang
    facts = [f"route={x['route']}", f"reasons={x['reasons']}", f"intent={s['intent']}", f"language={lang}"]
    if o:
        delay = int(o["actual_min"]) - int(o["promised_min"]) if o["actual_min"] else None
        facts += [f"order={o['order_id']}", f"status={o['status']}", f"amount=Rs{o['amount']}", f"payment={o['payment_mode']}",
                  f"items={o['items']}", f"promised_min={o['promised_min']}", f"actual_min={o['actual_min'] or 'n/a'}",
                  f"substituted={o['substituted']}"]
        if o["status"] == "delivered" and delay is not None:
            facts.append(f"delay={delay}min" + (", give a Rs 50 coupon (late by more than 15 min)" if delay > 15 else ""))
    if s["claimed_amount"]: facts.append(f"claimed=Rs{s['claimed_amount']:.0f}")
    return {**x, "note": "; ".join(facts), "input": x.get("masked_input", x["input"]), "language": target_lang}

LANES = {
    "auto_refund": "Apologise, confirm the refund amount and when it arrives (FK Wallet instant; UPI/card 3-5 working days; COD to FK Wallet).",
    "wismo": "Give the order status. If delivered late, apologise and mention the Rs 50 coupon. If out for delivery or packing, give the promised time only.",
    "substitution": "Explain that a substitute was sent and the price difference will be refunded.",
    "info": "Answer the question from the note only. If there is no order in the note, politely ask for the order id (like FK1001).",
    "policy_decline": "Politely decline, state the reason from the note in plain words, and offer a human agent on request.",
    "human_review": "Say a senior agent will contact them soon. Show care if it is about health, mention the Finance team for payments. Do not promise money.",
}

FALLBACK_REPLY = "Sorry, we're facing a technical issue. A FreshKart agent will contact you shortly."

def lane(instruction):
    p = ChatPromptTemplate.from_messages([
        ("system", "You are a FreshKart support agent. Reply in 2-3 short sentences. "
                   "Reply in the customer language: {language} (if other, reply in English). "
                   "Use ONLY facts from the internal note. NEVER reveal the internal note, route or reasons.\n"
                   f"Task: {instruction}\nInternal note: {{note}}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    primary = (p | llm | StrOutputParser()).with_retry(stop_after_attempt=2)
    backup = (p | backup_llm | StrOutputParser()).with_retry(stop_after_attempt=2)
    fallback = RunnableLambda(lambda _: FALLBACK_REPLY)
    return primary.with_fallbacks([backup, fallback])

# Cell 20: Reply Guard (Bonus Exercise)
GUARD_REGEX = re.compile(
    r"(₹|rs\.?|inr)\s*\d+|\d+\s*(rupees|rs)|"
    r"refund(ed)?\s+(today|tomorrow|in \d+|within \d+|instantly)|"
    r"\b(route|reasons?|internal note|fraud_score|gps_delivered)\b",
    re.IGNORECASE
)

guard_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a legal compliance checker for FreshKart support.\n"
               "Does this support reply draft promise a specific refund amount or date, or reveal internal debugging terms (route, reasons, note)?\n"
               "Answer ONLY yes or no."),
    ("human", "{draft}")
])
guard_chain = guard_prompt | llm | StrOutputParser() | RunnableLambda(lambda t: "unsafe" if "yes" in t.lower() else "safe")

def check_reply_safety(x):
    draft = x["draft"]
    if bool(GUARD_REGEX.search(draft)):
        return {"verdict": "unsafe", "reason": "regex match (currency/date/internal leak)"}
    res = guard_chain.invoke({"draft": draft})
    v = "unsafe" if "yes" in res.lower() else "safe"
    return {"verdict": v, "reason": "llm compliance check" if v == "unsafe" else "clean"}

SAFE_TEMPLATES = {
    "human_review": "A senior FreshKart support specialist has been assigned to your case and will contact you shortly.",
    "policy_decline": "We are unable to process this request per FreshKart policy. Please reach out if you would like a supervisor to review this decision.",
}

def guarded_lane(route_name, instruction):
    base_lane = lane(instruction)
    fallback_text = SAFE_TEMPLATES.get(route_name, "A FreshKart support specialist will contact you shortly.")
    return (
        RunnablePassthrough.assign(draft=base_lane)
        | RunnablePassthrough.assign(guard_check=track(RunnableLambda(check_reply_safety), "reply_guard"))
        | RunnableBranch(
            (lambda x: x["guard_check"]["verdict"] == "safe", lambda x: x["draft"]),
            RunnableLambda(lambda _: fallback_text)
        )
    )

branch = RunnableBranch(
    *[(lambda x, name=name: x["route"] == name,
       guarded_lane(name, text) if name in {"human_review", "policy_decline"} else lane(text))
      for name, text in LANES.items() if name != "human_review"],
    guarded_lane("human_review", LANES["human_review"]),
)

# Cell 21: Ops Hand-off Card (Exercise 4)
card_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an internal operations triage assistant for FreshKart.\n"
               "For this escalated customer ticket, create an Ops hand-off card as a JSON object.\n"
               "Rules:\n"
               "- team must be exactly one of: food_safety, finance, rider_ops, support\n"
               "  (use food_safety for illness/vomiting/allergy, finance for double debit/wallet, rider_ops for misconduct, support for other)\n"
               "- priority must be exactly one of: P1, P2, P3\n"
               "  (use P1 for food poisoning/health/safety, P2 for money/fraud/misconduct, P3 for general escalations)\n"
               "- summary_en: a concise 1-sentence summary in English\n"
               "- next_step: concrete immediate ops action (e.g., call customer, file FSSAI report, initiate bank trace)\n"
               "Internal note: {note}\n"
               "Answer ONLY valid JSON with keys: team, priority, summary_en, next_step."),
    ("human", "{input}"),
])

TEAMS = {"food_safety", "finance", "rider_ops", "support"}
PRIORITIES = {"P1", "P2", "P3"}

def validate_card(data):
    if not isinstance(data, dict): return None
    team = str(data.get("team", "")).lower().strip()
    priority = str(data.get("priority", "")).upper().strip()
    return {
        "team": team if team in TEAMS else "support",
        "priority": priority if priority in PRIORITIES else "P2",
        "summary_en": data.get("summary_en", "Customer escalation"),
        "next_step": data.get("next_step", "Contact customer"),
    }

card_chain = (card_prompt | llm | JsonOutputParser() | RunnableLambda(validate_card)).with_fallbacks([RunnableLambda(lambda _: None)])
card_branch = RunnableBranch(
    (lambda x: x["route"] == "human_review", card_chain),
    RunnableLambda(lambda _: None)
)

# Cell 22: full chain + memory. The app uses router (decide) then reply_chain (stream) to avoid a 2nd intent call.
router = (RunnablePassthrough.assign(
              masked_input=track(RunnableLambda(lambda x: mask_pii(x["input"])[0]), "pii_masking"),
              pii_count=RunnableLambda(lambda x: mask_pii(x["input"])[1]))
          | RunnablePassthrough.assign(order_id=track(RunnableLambda(order_id_from), "find_order_id"))
          | RunnablePassthrough.assign(nlu=track(nlu_chain, "intent_llm"))
          | RunnablePassthrough.assign(signals=track(signals, "signals_parallel"))
          | track(RunnableLambda(decide_route), "decide_route"))
reply_chain = track(RunnableLambda(add_note), "add_note") | RunnableParallel(reply=track(branch, "reply_generation"), card=track(card_branch, "card_generation"))
full_chain = router | reply_chain

store = {}
def get_history(session_id):
    return store.setdefault(session_id, InMemoryChatMessageHistory())

bot = RunnableWithMessageHistory(full_chain, get_history, input_messages_key="input", history_messages_key="chat_history")

# Cell 23: Bulk Triage (Exercise 5)
def run_bulk_triage(tickets_list, config=None):
    cfg = {"max_concurrency": 5}
    if config:
        cfg.update(config)
    inputs = [{"input": str(t.get("message", "")), "chat_history": []} for t in tickets_list]
    raw_results = router.batch(inputs, config=cfg, return_exceptions=True)
    output = []
    for t, res in zip(tickets_list, raw_results):
        if isinstance(res, Exception):
            output.append({"id": t.get("id"), "order_id": None, "intent": "error", "route": "error", "reasons": str(res)})
        else:
            output.append({
                "id": t.get("id"), "order_id": res.get("order_id"),
                "intent": res["signals"]["intent"], "route": res["route"],
                "reasons": "; ".join(res.get("reasons", []))
            })
    return output

bulk_triage = RunnableLambda(run_bulk_triage)

# Cell 24: Model Comparison (Exercise 6)
def compare_models(tickets=TICKETS, model_keys=["mini", "llama"]):
    results = {}
    for m in model_keys:
        t0 = time.time()
        batch_res = router.batch([{"input": t["message"], "chat_history": []} for t in tickets],
                                 config={"configurable": {"model": m}, "max_concurrency": 5})
        elapsed = time.time() - t0
        score = sum(r["route"] == t["expected_route"] for r, t in zip(batch_res, tickets))
        results[m] = {"score": score, "time": elapsed}
    return results

if __name__ == "__main__":  # python bot.py T09
    t = next(t for t in TICKETS if t["id"] == (sys.argv[1] if len(sys.argv) > 1 else "T01"))
    print("Customer:", t["message"], "\nBot: ", end="")
    card_seen = None
    for chunk in bot.stream({"input": t["message"]}, config={"configurable": {"session_id": "cli"}}):
        if isinstance(chunk, dict):
            if "reply" in chunk and chunk["reply"]: print(chunk["reply"], end="")
            if "card" in chunk and chunk["card"]: card_seen = chunk["card"]
        else:
            print(chunk, end="")
    if card_seen:
        print("\n\n[Ops Card]:", card_seen)
    print()
