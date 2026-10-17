import os, time, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="FreshKart Support Desk", page_icon="🛒", layout="wide")
st.title("FreshKart Support Desk")
load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY is missing. Copy .env.example to .env, add your key, and restart.")
    st.stop()
from bot import (TICKETS, orders, customers, inventory, riders, router, reply_chain, get_history, bulk_triage,
                 MODELS, evaluate, SIGNAL_STATIONS, trace_table,
                 get_order, get_customer, order_id_from)

ss = st.session_state
ss.setdefault("sid", uuid.uuid4().hex[:8])
ss.setdefault("messages", [])  # rendered chat: role, content, route, reasons, signals
ss.setdefault("log", [])       # one row per routed ticket (for the Ops tab)
COLORS = {"auto_refund": "green", "wismo": "blue", "substitution": "violet", "info": "gray",
          "policy_decline": "orange", "human_review": "red"}

def why(m):
    with st.expander("Why this answer?"):
        st.markdown(f"Route: :{COLORS[m['route']]}[**{m['route']}**]")
        st.markdown("Reasons: " + "; ".join(m["reasons"]))
        st.markdown(f"Language: **{m['signals'].get('language', 'english')}**")
        st.markdown(f"Model chosen: `{m.get('model', 'mini')}` · intent answered by: `{m['signals'].get('intent_model', 'n/a')}`")
        if m.get("pii_count"):
            st.markdown(f":shield: PII masked: **{m['pii_count']} items** (before any prompt)")
        st.json(m["signals"])
        if m.get("card"):
            st.markdown("**Hand-off card** (for the senior agent)")
            st.json(m["card"])
        if m.get("trace"):
            st.markdown("**Pipeline tracker**")
            st.code(m["trace"], language="text")

def tracker_text(trace, decision=None):
    """Ex7 display only: one line per station from the list the pipeline's listeners filled."""
    rows, lines = trace_table(trace), []
    for r in rows:
        if r["seconds"] is None:
            lines.append(f"… {r['station']:<20} running")
            continue
        routed = f"  -> {decision['route']} ({'; '.join(decision['reasons'])})" if decision and r["station"] == "decide_route" else ""
        lines.append(f"✓ {r['station']:<20} {r['seconds']:5.2f}s  at {r['at_s']:.2f}s{routed}")
    total = next((r["seconds"] for r in rows if r["station"] == "router" and r["seconds"] is not None), None)
    if total:
        signals_sum = sum(r["seconds"] or 0 for r in rows if r["station"] in SIGNAL_STATIONS)
        lines.append(f"sum of signal stations {signals_sum:.2f}s vs router total {total:.2f}s"
                     + ("  (they overlapped: RunnableParallel)" if signals_sum > total else ""))
    return "\n".join(lines)

def run_tracked(fn, trace, box):
    """Run the pipeline on a worker thread and redraw the tracker from Streamlit's thread until it is done."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        while not future.done():
            box.code(tracker_text(trace), language="text")
            time.sleep(0.1)
    return future.result()

def reply_text(chunks, out):
    """reply_chain streams {"reply": text} chunks and one {"card": ...} chunk: show the text, keep the card."""
    for chunk in chunks:
        if chunk.get("reply"): yield chunk["reply"]
        if "card" in chunk: out["card"] = chunk["card"]

# Sidebar
sample = None
with st.sidebar:
    st.header("About")
    st.write("Rules-based support bot: signals in parallel, routing in plain Python, then a streamed reply.")
    model = st.selectbox("Model", list(MODELS), format_func=lambda k: f"{k} ({MODELS[k]})")
    cfg = {"configurable": {"model": model}}  # Ex6: same chains, the model is picked per request
    tracker_on = st.toggle("Live pipeline tracker", value=True)  # Ex7: off = no trace list, same pipeline
    choice = st.selectbox("Try a sample ticket", TICKETS, format_func=lambda t: f"{t['id']}: {' '.join(t['message'].split()[:6])}...")
    if st.button("Send"):
        sample = choice["message"]
    if st.button("New chat"):
        ss.sid, ss.messages = uuid.uuid4().hex[:8], []
        st.rerun()

msg = st.chat_input("Write to FreshKart support...") or sample
tab_chat, tab_ops, tab_data, tab_how = st.tabs(["Chat", "Ops queue", "Data", "How it works"])

with tab_chat:
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["role"] == "assistant": why(m)
    if msg:
        with st.chat_message("user"):
            st.write(msg)
        history = get_history(ss.sid)
        with st.chat_message("assistant"):
            box = status = None
            if tracker_on:
                trace = []
                run_cfg = {**cfg, "configurable": {**cfg["configurable"], "trace": trace}}
                with st.status("Processing the ticket on the belt...", expanded=True) as status:
                    box = st.empty()
                    decision = run_tracked(lambda: router.invoke({"input": msg, "chat_history": history.messages},
                                                                 config=run_cfg), trace, box)
                    box.code(tracker_text(trace, decision) + "\n… writing reply", language="text")
            else:
                run_cfg = cfg
                with st.spinner("Checking order, customer and policy..."):
                    decision = router.invoke({"input": msg, "chat_history": history.messages}, config=cfg)
            out = {}
            reply = st.write_stream(reply_text(reply_chain.stream(decision, config=run_cfg), out))
            card = out.get("card")
            if tracker_on:
                box.code(tracker_text(trace, decision), language="text")
                status.update(label=f"Done: {decision['route']}", state="complete", expanded=False)
            m = {"role": "assistant", "content": reply, "route": decision["route"],
                 "reasons": decision["reasons"], "signals": decision["signals"],
                 "pii_count": decision["pii_count"], "card": card, "model": model,
                 "trace": tracker_text(trace, decision) if tracker_on else None}
            why(m)
        history.add_user_message(msg)
        history.add_ai_message(reply)
        ss.messages += [{"role": "user", "content": msg}, m]
        ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "order_id": decision["order_id"],
                       "intent": decision["signals"]["intent"], "language": decision["signals"]["language"],
                       "route": decision["route"],
                       "reasons": "; ".join(decision["reasons"]), "message": msg,
                       "team": (card or {}).get("team"), "priority": (card or {}).get("priority"),
                       "summary": (card or {}).get("summary_en")})

with tab_ops:
    cols = st.columns(len(COLORS))
    for col, route in zip(cols, COLORS):
        col.metric(route, sum(r["route"] == route for r in ss.log))
    queue = [r for r in ss.log if r["route"] == "human_review"]
    st.subheader("Human review queue")
    if queue:
        st.dataframe(pd.DataFrame(queue).drop(columns="route"), use_container_width=True)
    else:
        st.info("No tickets sent to a human yet in this session.")

    st.subheader("Bulk triage (route only, no replies)")
    upload = st.file_uploader("Overnight tickets CSV with columns id, message", type="csv")
    rows = None
    if upload is not None and st.button("Triage"):
        df = pd.read_csv(upload, dtype=str, keep_default_na=False)  # empty cells stay "" (not NaN)
        if {"id", "message"} <= set(df.columns):
            rows = df.to_dict("records")
        else:
            st.error("The CSV needs the columns id and message.")
    if st.button("Triage the 18 sample tickets"):
        rows = TICKETS
    if rows is not None:
        t0 = time.perf_counter()
        result = pd.DataFrame(bulk_triage.invoke(rows, config=cfg))
        ss.bulk = {"result": result, "seconds": time.perf_counter() - t0,
                   "expected": [r.get("expected_route") for r in rows]}
    if "bulk" in ss:
        result = ss.bulk["result"]
        st.caption(f"{len(result)} tickets routed in {ss.bulk['seconds']:.1f}s (router.batch, max_concurrency=5)")
        if all(ss.bulk["expected"]):
            st.caption(f"Matches expected_route: {sum(result.route == ss.bulk['expected'])}/{len(result)}")
        st.dataframe(result.route.value_counts().rename("tickets"))
        st.dataframe(result, use_container_width=True)

    st.subheader("Compare models (18 sample tickets, routing only)")
    if st.button("Compare models"):
        # one model at a time, so the timings do not disturb each other
        ss.compare = pd.DataFrame(evaluate.batch(list(MODELS), config={"max_concurrency": 1}))
    if "compare" in ss:
        st.dataframe(ss.compare, use_container_width=True)

with tab_data:
    for name, df in [("orders", orders), ("customers", customers), ("inventory", inventory), ("riders", riders)]:
        st.subheader(name)
        st.dataframe(df, use_container_width=True)

with tab_how:
    st.code("""
 {input, chat_history}   + config={"configurable": {"model": "mini" | "llama" | "strong"}}  (Ex6)
        |
 RunnablePassthrough.assign(masked_input, pii_count)   Ex1: mask phone/email/UPI/card for prompts
        |
 RunnableLambda(find_order_id)        regex: FK1001 / fk1001 / ord_1001
        |
 RunnablePassthrough.assign(signals = RunnableParallel(
        intent (LLM) | language (LLM, Ex2) | order | customer | inventory | rider | safety | claimed_amount ))
        |
 RunnableLambda(unpack_intent)        Ex3: intent + which model answered (retry -> backup model -> "unclear")
        |
 RunnableLambda(decide_route)         plain Python rules, no LLM
                                      (Ex7: every station is named and timed by listeners -> live tracker)
        |
 RunnableBranch: auto_refund | wismo | substitution | info | policy_decline | human_review
        |           (Bonus: policy_decline + human_review drafts -> reply guard: regex, then LLM yes/no
        |            -> draft if safe, else a fixed safe template; these two lanes no longer stream)
        |
 StrOutputParser -> streamed reply in the customer's language   (memory: RunnableWithMessageHistory)
        +  in parallel (Ex4): human_review only -> card_prompt | llm | JsonOutputParser -> hand-off card
""", language="text")
    st.table(pd.DataFrame([
        ("Cell 2 Setup", "get_llm, read", "Data tab"),
        ("Cell 10 RunnableLambda", "find_order_id", "Order info in sidebar"),
        ("Cell 12 Lookups", "get_order, get_customer, check_inventory, get_rider", "Why this answer? (signals)"),
        ("Cell 14 assign + memory", "order_id_from", "Order recovered from earlier chat turns"),
        ("Cell 16 RunnableParallel", "signals, parse_intent", "Why this answer? (signals)"),
        ("Cell 18 decide_route", "decide_route", "Route badge, Ops queue, metrics"),
        ("Cell 20 RunnableBranch", "add_note, branch, reply_chain", "Streamed reply in Chat"),
        ("Cell 22 Memory", "full_chain, store, get_history", "New chat button"),
        ("Cell 24 Evaluation", "router", "Sample ticket + Send"),
    ], columns=["Notebook cell", "bot.py", "App widget"]))

# Sidebar order panel for the latest message
with st.sidebar:
    if ss.messages:
        last = [m for m in ss.messages if m["role"] == "user"][-1]["content"]
        oid = order_id_from({"input": last, "chat_history": get_history(ss.sid).messages})
        if oid and get_order(oid):
            o, c = get_order(oid), get_customer(oid)
            st.divider()
            st.caption(f"{oid}: {o['status']}, Rs {o['amount']}, {o['payment_mode']}")
            st.caption(f"{c['name']} ({c['tier']}), refunds 30d: {c['refunds_30d']}")
