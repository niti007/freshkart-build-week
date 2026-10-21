import os, uuid
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
from bot import (TICKETS, orders, customers, inventory, riders, router, reply_chain, get_history,
                 get_order, get_customer, order_id_from, bulk_triage, MODELS, tracked_router)

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
        if m.get("pii_masked_count"):                                      # Ex 1: stretch – PII badge
            st.markdown(f"🛡️ PII masked: **{m['pii_masked_count']}** item(s)")
        if m.get("card"):                                                   # Ex 4: hand-off card
            st.markdown("📋 **Hand-off card:**")
            st.json(m["card"])
        st.json(m["signals"])

# Sidebar
sample = None
with st.sidebar:
    st.header("About")
    st.write("Rules-based support bot: signals in parallel, routing in plain Python, then a streamed reply.")
    # Ex 6: Model switcher – sidebar select box, config passed per request.
    model_choice = st.selectbox("Model", list(MODELS.keys()),
                                format_func=lambda k: f"{k} ({MODELS[k]})")
    choice = st.selectbox("Try a sample ticket", TICKETS, format_func=lambda t: f"{t['id']}: {' '.join(t['message'].split()[:6])}...")
    if st.button("Send"):
        sample = choice["message"]
    if st.button("New chat"):
        ss.sid, ss.messages = uuid.uuid4().hex[:8], []
        st.rerun()
    # Ex 6: Compare models button
    if st.button("Compare models"):
        import time as _time
        st.write("Running 18-ticket eval on each model...")
        for key, name in MODELS.items():
            cfg = {"configurable": {"model": key}}
            t0 = _time.time()
            res = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS], config=cfg)
            elapsed = _time.time() - t0
            score = sum(r["route"] == t["expected_route"] for r, t in zip(res, TICKETS))
            st.write(f"**{key}** ({name}): {score}/18 in {elapsed:.1f}s")

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
        # Ex 6: pass model choice via config on every call (chain is not rebuilt).
        model_config = {"configurable": {"model": model_choice}}
        with st.chat_message("assistant"):
            # Ex 7: tracked_router returns (decision, events) with timing for each station.
            with st.status("Pipeline steps...", expanded=True) as pipeline_status:
                decision, events = tracked_router({"input": msg, "chat_history": history.messages}, config=model_config)
                if events:
                    sorted_events = sorted(events, key=lambda e: e["start"])
                    for ev in sorted_events:
                        secs = (ev["end"] - ev["start"]).total_seconds()
                        extra = ""
                        if ev["name"] == "decide_route":
                            extra = f"  → {decision['route']} ({decision['reasons'][0]})"
                        pipeline_status.text(f"✓ {ev['name']:18s} {secs:.2f}s{extra}")
                    t_min = min(e["start"] for e in events)
                    t_max = max(e["end"] for e in events)
                    wall = (t_max - t_min).total_seconds()
                    parts_sum = sum((e["end"] - e["start"]).total_seconds() for e in events)
                    pipeline_status.text(f"Wall: {wall:.2f}s | Sum: {parts_sum:.2f}s (parallel!)")
                pipeline_status.update(label="Pipeline complete ✓", state="complete", expanded=False)
            # Ex 4: reply_chain now streams {"reply": str_chunk, "card": dict|None} chunks.
            # Display reply tokens live; capture card when it arrives.
            placeholder = st.empty()
            reply_text, card = "", None
            for chunk in reply_chain.stream(decision, config=model_config):
                if "reply" in chunk and chunk["reply"]:
                    reply_text += chunk["reply"]
                    placeholder.markdown(reply_text + "▌")
                if "card" in chunk:
                    card = chunk["card"]
            placeholder.markdown(reply_text)
            reply = reply_text
            m = {"role": "assistant", "content": reply, "route": decision["route"],
                 "reasons": decision["reasons"], "signals": decision["signals"],
                 "pii_masked_count": decision.get("pii_masked_count", 0),
                 "card": card}                                              # Ex 4: card in message
            why(m)
        history.add_user_message(msg)
        history.add_ai_message(reply)
        ss.messages += [{"role": "user", "content": msg}, m]
        log_entry = {"time": datetime.now().strftime("%H:%M:%S"), "order_id": decision["order_id"],
                     "intent": decision["signals"]["intent"],
                     "language": decision["signals"].get("language", "english"),  # Ex 2
                     "route": decision["route"],
                     "reasons": "; ".join(decision["reasons"]), "message": msg}
        if card:                                                            # Ex 4: card columns in log
            log_entry.update({"team": card.get("team", ""), "priority": card.get("priority", ""),
                              "summary": card.get("summary_en", "")})
        ss.log.append(log_entry)

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

    # Ex 5: Bulk triage – upload a CSV, route all tickets at once (no replies).
    st.divider()
    st.subheader("Bulk triage")
    uploaded = st.file_uploader("Upload CSV (columns: id, message)", type="csv")
    if uploaded and st.button("Triage"):
        import time
        df_upload = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
        messages = df_upload.to_dict("records")
        with st.spinner(f"Triaging {len(messages)} tickets..."):
            t0 = time.time()
            rows = bulk_triage(messages)
            elapsed = time.time() - t0
        result_df = pd.DataFrame(rows)
        st.success(f"Triaged {len(rows)} tickets in {elapsed:.1f}s")
        # Count per route
        route_counts = result_df["route"].value_counts()
        st.dataframe(route_counts.rename("count").to_frame(), use_container_width=False)
        st.dataframe(result_df, use_container_width=True)

with tab_data:
    for name, df in [("orders", orders), ("customers", customers), ("inventory", inventory), ("riders", riders)]:
        st.subheader(name)
        st.dataframe(df, use_container_width=True)

with tab_how:
    st.code("""
 {input, chat_history}
        |
 RunnableLambda(find_order_id)        regex: FK1001 / fk1001 / ord_1001
        |
 RunnablePassthrough.assign(signals = RunnableParallel(
        intent (LLM) | order | customer | inventory | rider | safety | claimed_amount ))
        |
 RunnableLambda(decide_route)         plain Python rules, no LLM
        |
 RunnableBranch: auto_refund | wismo | substitution | info | policy_decline | human_review
        |
 StrOutputParser -> streamed reply     (memory: RunnableWithMessageHistory)
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
        oid = order_id_from({"_raw_input": last, "chat_history": get_history(ss.sid).messages})
        if oid and get_order(oid):
            o, c = get_order(oid), get_customer(oid)
            st.divider()
            st.caption(f"{oid}: {o['status']}, Rs {o['amount']}, {o['payment_mode']}")
            st.caption(f"{c['name']} ({c['tier']}), refunds 30d: {c['refunds_30d']}")
