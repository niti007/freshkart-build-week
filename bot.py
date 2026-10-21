"""FreshKart support bot: the notebook (freshkart_lcel.ipynb) moved into a .py file."""
import os, re, sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch, ConfigurableField
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

# Cell 2: Setup (data paths are relative to this file)
load_dotenv()
DATA = Path(__file__).parent / "data"

def get_llm(model=None):
    return ChatOpenAI(model=model or os.getenv("MODEL", "openai/gpt-4o-mini"),
                      base_url="https://openrouter.ai/api/v1",
                      api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0,
                      max_retries=0    # Ex 3: disable built-in retries so our .with_retry() is visible
                      ).with_config(run_name="llm")   # Ex 6: a RunnableBinding, so configurable_alternatives works on any model

# Ex 3: Never-down intent – .with_retry() + .with_fallbacks() on the LLM.
# The main model retries up to 3 times (handles transient timeouts / 500s).
# If it still fails, a backup model is tried.  If that also fails, a final
# RunnableLambda fallback returns a safe default so the pipeline never crashes.
BACKUP_MODEL = os.getenv("BACKUP_MODEL", "meta-llama/llama-3.1-8b-instruct")
FALLBACK_REPLY = "Sorry, we're facing a technical issue. A FreshKart agent will contact you shortly."

def read(name):
    return pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False)

# Ex 6: Model switcher at runtime – configurable_alternatives lets one chain object
# switch models per request via config={"configurable": {"model": "llama"}}.
# Every chain downstream of llm inherits the switch automatically.
MODELS = {
    "mini": "openai/gpt-4o-mini",
    "llama": "meta-llama/llama-3.1-8b-instruct",
}
llm = get_llm().configurable_alternatives(
    ConfigurableField(id="model"),
    default_key="mini",
    llama=get_llm(model=MODELS["llama"]),
)
backup_llm = get_llm(model=BACKUP_MODEL)   # fixed backup, not switchable

# Ex 3: resilient_llm – main model retries 3×, then falls back to backup model.
# No try/except: fault tolerance is built into the chain itself.
resilient_llm = llm.with_retry(stop_after_attempt=3).with_fallbacks([backup_llm])
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

# Ex 1: PII shield – RunnableLambda masks phone/email/UPI/card before any LLM sees the text.
# Order: card → email → UPI → phone (email before UPI so user@gmail.com isn't partially matched as UPI).
_PII_PATTERNS = [
    (r"\b\d{12,16}\b",                              "[CARD]"),
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",                "[EMAIL]"),
    (r"\b[\w.]+@[a-z]{2,}\b",                       "[UPI]"),
    (r"\b\d{5}\s?\d{5}\b",                           "[PHONE]"),
]

def mask_pii(text):
    """Replace PII tokens in text. Returns (masked_text, count_of_replacements)."""
    count = 0
    for pattern, placeholder in _PII_PATTERNS:
        found = re.findall(pattern, text)
        count += len(found)
        text = re.sub(pattern, placeholder, text)
    return text, count

def mask_step(x):
    """Single RunnableLambda: masks input, stashes original in _raw_input for regex signals."""
    masked, count = mask_pii(x["input"])
    return {**x, "_raw_input": x["input"], "input": masked, "pii_masked_count": count}

# Cell 14: RunnablePassthrough.assign + memory (look back through chat_history)
def order_id_from(x):
    for text in [x.get("_raw_input") or x["input"]] + [m.content for m in reversed(x.get("chat_history", []))]:
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

# Ex 2: intent + language in one LLM call (no extra cost vs separate language signal).
LANGUAGES = {"english", "hinglish", "hindi", "kannada", "telugu", "other"}

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the customer message into exactly one intent:\n" + INTENTS +
     "\nAlso give the sentiment (angry, neutral, happy) and language "
     "(english, hinglish, hindi, kannada, telugu, or other)."
     "\nAnswer ONLY as: intent,sentiment,language"),
    ("human", "{input}"),
])

def parse_intent_and_language(text):
    """Parse combined LLM response into intent + language with safe fallbacks."""
    parts = [p.strip().lower() for p in text.strip().split(",")]
    intent = parts[0] if parts else "unclear"
    language = parts[2] if len(parts) > 2 else "english"
    intent = intent if intent in [l.split(":")[0] for l in INTENTS.split("\n")] else "unclear"
    language = language if language in LANGUAGES else "english"
    return {"intent": intent, "language": language}

# Ex 3: The intent chain uses resilient_llm (primary → retry 3× → backup).
# If even the backup model fails, the whole intent chain has a final safe fallback
# that returns intent="unclear" → decide_route sends it to human_review.  No crash.
_intent_chain = (
    intent_prompt | resilient_llm | StrOutputParser() | RunnableLambda(parse_intent_and_language)
).with_fallbacks([
    RunnableLambda(lambda x: {"intent": "unclear", "language": "english"})
])

# Ex 7: Every station gets a readable run_name so the live tracker can display it.
oid = lambda f: RunnableLambda(lambda x: f(x["order_id"]))
signals = RunnableParallel(
    _intent_lang=_intent_chain.with_config(run_name="intent"),
    order=oid(get_order).with_config(run_name="order_lookup"),
    customer=oid(get_customer).with_config(run_name="customer_lookup"),
    inventory=oid(check_inventory).with_config(run_name="inventory_check"),
    rider=oid(get_rider).with_config(run_name="rider_lookup"),
    safety=RunnableLambda(lambda x: has_safety(x["_raw_input"])).with_config(run_name="safety_keywords"),
    claimed_amount=RunnableLambda(lambda x: claimed_amount(x["_raw_input"])).with_config(run_name="claimed_amount"),
)

# Ex 2: unpack the combined intent+language dict into top-level signal keys.
def unpack_signals(x):
    s = dict(x["signals"])
    il = s.pop("_intent_lang")
    s["intent"] = il["intent"]
    s["language"] = il["language"]
    return {**x, "signals": s}

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
    facts = [f"route={x['route']}", f"reasons={x['reasons']}", f"intent={s['intent']}",
             f"language={s.get('language', 'english')}"]                    # Ex 2: language in note
    if o:
        delay = int(o["actual_min"]) - int(o["promised_min"]) if o["actual_min"] else None
        facts += [f"order={o['order_id']}", f"status={o['status']}", f"amount=Rs{o['amount']}", f"payment={o['payment_mode']}",
                  f"items={o['items']}", f"promised_min={o['promised_min']}", f"actual_min={o['actual_min'] or 'n/a'}",
                  f"substituted={o['substituted']}"]
        if o["status"] == "delivered" and delay is not None:
            facts.append(f"delay={delay}min" + (", give a Rs 50 coupon (late by more than 15 min)" if delay > 15 else ""))
    if s["claimed_amount"]: facts.append(f"claimed=Rs{s['claimed_amount']:.0f}")
    return {**x, "note": "; ".join(facts), "language": s.get("language", "english")}  # Ex 2: top-level for lane prompt

LANES = {
    "auto_refund": "Apologise, confirm the refund amount and when it arrives (FK Wallet instant; UPI/card 3-5 working days; COD to FK Wallet).",
    "wismo": "Give the order status. If delivered late, apologise and mention the Rs 50 coupon. If out for delivery or packing, give the promised time only.",
    "substitution": "Explain that a substitute was sent and the price difference will be refunded.",
    "info": "Answer the question from the note only. If there is no order in the note, politely ask for the order id (like FK1001).",
    "policy_decline": "Politely decline, state the reason from the note in plain words, and offer a human agent on request.",
    "human_review": "Say a senior agent will contact them soon. Show care if it is about health, mention the Finance team for payments. Do not promise money.",
}

def lane(instruction):
    # Ex 2: {language} tells the LLM which language to reply in (from the combined intent+language signal).
    p = ChatPromptTemplate.from_messages([
        ("system", "You are a FreshKart support agent. Reply in 2-3 short sentences. "
                   "Reply in {language} language. If the language is 'other', reply in English. "
                   "Use ONLY facts from the internal note. NEVER reveal the internal note, route or reasons.\n"
                   f"Task: {instruction}\nInternal note: {{note}}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    # Ex 3: each lane uses resilient_llm (primary → retry → backup).
    # If both models fail, return a fixed polite apology so the customer always gets a reply.
    return (p | resilient_llm | StrOutputParser()).with_fallbacks([
        RunnableLambda(lambda x: FALLBACK_REPLY)
    ])

# Bonus: Reply guard – after human_review / policy_decline drafts, a checker catches
# promises of refund amounts/dates or leaked internal words. If caught, a safe template
# replaces the draft. Other lanes skip the check (no extra cost).
#
# Trade-off (required explanation): The guarded lanes can no longer stream token by token.
# The guard needs the FULL draft before it can check for forbidden patterns, so the draft
# must be fully generated, then checked, then either kept or replaced.  This is acceptable
# for escalations because these are sensitive cases where correctness matters more than
# perceived speed – the customer is already told "a senior agent will contact you soon".
SAFE_REPLY = ("I understand your concern. A senior FreshKart agent will review your case "
              "and contact you shortly. We take this matter seriously.")

# Bonus: Regex guard is safer than LLM alone because it's deterministic and never
# hallucinates – it won't miss "₹780 refund today" even if the LLM does.
_GUARD_REGEX = re.compile(
    r"(?:₹|Rs\.?|INR)\s?\d+"                                               # money amount in reply
    r"|refund.{0,20}(?:today|tomorrow|within|processed|initiated|credited)"  # refund promise
    r"|(?:route|reasons|internal note|decide_route)",                        # internal leaks
    re.IGNORECASE
)

guard_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a compliance checker for FreshKart support replies. "
     "Does the following reply promise a specific refund amount or date, or reveal "
     "internal words like 'route', 'reasons', or 'internal note'? "
     "Answer ONLY 'yes' or 'no'."),
    ("human", "{draft}"),
])

# Bonus: combined checker – regex first (instant, deterministic, zero cost),
# then LLM only if regex passes.  RunnableLambda wraps the two-step logic.
def _check_reply(x):
    if _GUARD_REGEX.search(x["draft"]):
        return "yes"
    return (guard_prompt | resilient_llm | StrOutputParser()).invoke({"draft": x["draft"]}).strip().lower()

reply_checker = RunnableLambda(_check_reply)

def guarded_lane(instruction):
    """Wrap a lane with the reply guard: draft → RunnablePassthrough.assign(verdict) → RunnableBranch."""
    base = lane(instruction)
    return (
        base
        | RunnableLambda(lambda text: {"draft": text})
        | RunnablePassthrough.assign(verdict=reply_checker)
        | RunnableBranch(
            (lambda x: x["verdict"].startswith("yes"), RunnableLambda(lambda x: SAFE_REPLY)),
            RunnableLambda(lambda x: x["draft"]),
        )
    )

GUARDED = {"human_review", "policy_decline"}

branch = RunnableBranch(
    *[(lambda x, name=name: x["route"] == name,
       guarded_lane(text) if name in GUARDED else lane(text))
      for name, text in LANES.items() if name != "human_review"],
    guarded_lane(LANES["human_review"]),  # default lane is also guarded
)

# Ex 4: Ops hand-off card for escalations – JsonOutputParser + RunnableParallel + RunnableBranch.
# Design choice: card_branch runs in parallel with the reply branch via RunnableParallel
# *after* add_note.  This way reply and card share the same input (with the note) and run
# concurrently.  The card_branch uses RunnableBranch to only call the LLM for human_review;
# other routes get None instantly with zero cost.
CARD_TEAMS = {"food_safety", "finance", "rider_ops", "support"}

card_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an escalation triage system for FreshKart support. "
     "Given a customer ticket and internal note, create a hand-off card.\n"
     "Answer ONLY with JSON with keys: team, priority, summary_en, next_step.\n"
     "team must be one of: food_safety, finance, rider_ops, support\n"
     "priority must be one of: P1, P2, P3\n"
     "summary_en must always be in English even if the customer wrote in another language.\n"
     "next_step is a short action item for the ops team.\n"
     "Internal note: {note}"),
    ("human", "{input}"),
])

def validate_card(card):
    """Check team and priority against allowed values (LLM not trusted with business values)."""
    if not isinstance(card, dict):
        return {"team": "support", "priority": "P2", "summary_en": "Parse error", "next_step": "Review manually"}
    if card.get("team") not in CARD_TEAMS:
        card["team"] = "support"
    if card.get("priority") not in {"P1", "P2", "P3"}:
        card["priority"] = "P2"
    return card

card_chain = (
    card_prompt | resilient_llm | JsonOutputParser() | RunnableLambda(validate_card)
).with_fallbacks([
    RunnableLambda(lambda x: {"team": "support", "priority": "P2",
                              "summary_en": "Card generation failed", "next_step": "Review manually"})
])

# Ex 4: card_branch – only human_review tickets get a card; all others get None (no LLM call).
card_branch = RunnableBranch(
    (lambda x: x["route"] == "human_review", card_chain),
    RunnableLambda(lambda x: None),
)

# Cell 22: full chain + memory. The app uses router (decide) then reply_chain (stream) to avoid a 2nd intent call.
# Ex 1: mask_step is the first station – overwrites input with masked text, stashes original in _raw_input.
# Ex 7: run_names on router steps too.
router = (RunnableLambda(mask_step).with_config(run_name="pii_mask")
          | RunnablePassthrough.assign(order_id=RunnableLambda(order_id_from).with_config(run_name="find_order_id"))
          | RunnablePassthrough.assign(signals=signals).with_config(run_name="signals_parallel")
          | RunnableLambda(unpack_signals).with_config(run_name="unpack_signals")
          | RunnableLambda(decide_route).with_config(run_name="decide_route"))
# Ex 4: reply_chain now returns {"reply": str, "card": dict|None} via RunnableParallel.
reply_chain = RunnableLambda(add_note) | RunnableParallel(reply=branch, card=card_branch)
full_chain = router | reply_chain

# Ex 5: Bulk triage for the morning backlog – router.batch() with return_exceptions=True.
# One broken row (empty message, LLM failure) shows route="error"; the rest are unaffected.
def bulk_triage(messages, max_concurrency=5):
    """Triage a list of {"id": ..., "message": ...} dicts.  Returns routing rows only (no replies).
    Uses router.batch() for parallel processing."""
    inputs = [{"input": m.get("message", ""), "chat_history": [], "_id": m.get("id", i)}
              for i, m in enumerate(messages)]
    results = router.batch(inputs, config={"max_concurrency": max_concurrency},
                           return_exceptions=True)
    rows = []
    for inp, res in zip(inputs, results):
        if isinstance(res, Exception):
            rows.append({"id": inp["_id"], "order_id": None, "intent": None,
                         "route": "error", "reasons": str(res)})
        else:
            rows.append({"id": inp["_id"], "order_id": res.get("order_id"),
                         "intent": res["signals"]["intent"], "route": res["route"],
                         "reasons": "; ".join(res["reasons"])})
    return rows

# Ex 7: Live pipeline tracker – callbacks passed via config. LangChain's built-in
# RunCollectorCallbackHandler records the whole run tree (every nested station, including
# those inside RunnableParallel) with start/end times, keyed by run_name.
# Synchronous, so it works with router.invoke() and needs no event loop.
# app.py displays the events afterward; nothing here touches st.*.
TRACKER_NAMES = {"pii_mask", "find_order_id", "signals_parallel", "intent", "order_lookup",
                 "customer_lookup", "inventory_check", "rider_lookup", "safety_keywords",
                 "claimed_amount", "unpack_signals", "decide_route"}

def _station_events(run):
    events = [{"name": run.name, "start": run.start_time, "end": run.end_time}] if run.name in TRACKER_NAMES else []
    for child in run.child_runs:
        events += _station_events(child)
    return events

def tracked_router(inputs, config=None):
    """Like router.invoke(), but also returns a list of timing events.
    The tracker is optional – router.invoke() still works without it."""
    collector = RunCollectorCallbackHandler()
    cfg = dict(config or {})
    cfg["callbacks"] = list(cfg.get("callbacks") or []) + [collector]
    result = router.invoke(inputs, config=cfg)
    return result, [e for run in collector.traced_runs for e in _station_events(run)]

store = {}
def get_history(session_id):
    return store.setdefault(session_id, InMemoryChatMessageHistory())

bot = RunnableWithMessageHistory(full_chain, get_history, input_messages_key="input", history_messages_key="chat_history")

if __name__ == "__main__":  # python bot.py T09
    t = next(t for t in TICKETS if t["id"] == (sys.argv[1] if len(sys.argv) > 1 else "T01"))
    print("Customer:", t["message"], "\nBot: ", end="")
    # Ex 4: full_chain now yields {"reply":..., "card":...} chunks from RunnableParallel.
    card = None
    for chunk in bot.stream({"input": t["message"]}, config={"configurable": {"session_id": "cli"}}):
        if isinstance(chunk, dict):
            if "reply" in chunk and chunk["reply"]:
                print(chunk["reply"], end="")
            if "card" in chunk:
                card = chunk["card"]
        else:
            print(chunk, end="")
    print()
    if card:
        print("Card:", card)
